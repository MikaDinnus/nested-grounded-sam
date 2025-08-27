from groundingdino.util.inference import load_model, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import box_iou
from shapely.geometry import Polygon
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import json, os, time, sys

print("################################ STARTING EXPERIMENT FLAT ON", DATASET_NUMBER, "| UPSAMPLING:", UPSAMPLING, "################################")
CURRENT_DATASET = f"building_facade/ADE_train_{DATASET_NUMBER}"

# Mittelwerte (pro Lauf/Variante)
mean_value_boxes = 0.0
mean_value_segments = 0.0

# Ordner für Visuals/Ausgaben
base_dir = f"experiment_outputs/{DATASET_NUMBER}/{UPSAMPLING}"
os.makedirs(base_dir, exist_ok=True)

# Zeitmessung
_start_flat = time.process_time()

# ----------------- Helper -----------------
def convertcoords(boxes, width, height):
    # normierte (xc,yc,w,h) -> absolute XYXY in Pixeln
    x_center, y_center, w, h = boxes
    x1 = int((x_center - w/2) * width)
    y1 = int((y_center - h/2) * height)
    x2 = int((x_center + w/2) * width)
    y2 = int((y_center + h/2) * height)
    return x1, y1, x2, y2

def xcycwh_to_xyxy(b):
    xc, yc, w, h = b
    return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]

def bbox_from_polygon(x, y):
    return [min(x), min(y), max(x), max(y)]

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def find_objects(data, search_terms, return_polygon=False):
    results = []
    objects = data["annotation"].get("object", [])
    for obj in objects:
        name = obj.get("name", "").lower().strip()
        raw_name = obj.get("raw_name", "").lower().strip()
        poly = obj.get("polygon", {})
        x = poly.get("x", [])
        y = poly.get("y", [])
        found = False
        for term in search_terms:
            t = term.lower().strip()
            if t == name or t == raw_name:
                found = True
            else:
                if t in name.split() or t in raw_name.split():
                    found = True
                elif (name.startswith(t) and (len(name)==len(t) or name[len(t)] in [' ','-','_'])):
                    found = True
                elif (raw_name.startswith(t) and (len(raw_name)==len(t) or raw_name[len(t)] in [' ','-','_'])):
                    found = True
            if found: break
        if found and x and y:
            entry = {
                "id": obj.get("id"),
                "name": obj.get("name"),
                "raw_name": obj.get("raw_name"),
                "bbox": bbox_from_polygon(x, y)
            }
            if return_polygon:
                entry["polygon"] = {"x": x, "y": y}
            results.append(entry)
    return results

def getGTBoxes():
    data = load_json(f"{CURRENT_DATASET}.json")
    found = find_objects(data, VOCAB_GROUNDTRUTH, return_polygon=False)
    return [o["bbox"] for o in found]

def getGTSegments():
    data = load_json(f"{CURRENT_DATASET}.json")
    found = find_objects(data, VOCAB_GROUNDTRUTH, return_polygon=True)
    return [o["polygon"] for o in found]

def rescale_boxes_xyxy(boxes, scale):
    if scale == 1.0:
        return [[float(a) for a in b] for b in boxes]
    out = []
    for x1,y1,x2,y2 in boxes:
        out.append([x1*scale, y1*scale, x2*scale, y2*scale])
    return out

def rescale_segments(segs, scale):
    if scale == 1.0:
        return [{"x":[float(xx) for xx in s["x"]], "y":[float(yy) for yy in s["y"]]} for s in segs]
    out = []
    for s in segs:
        out.append({"x": [xx*scale for xx in s["x"]], "y": [yy*scale for yy in s["y"]]})
    return out

def dino(image_chw, caption, box_thr=0.35, text_thr=0.25):
    # GroundingDINO: Eingabe CHW float[0,1]
    boxes, logits, phrases = predict(
        model=grounding_dino_model,
        image=image_chw,
        caption=caption,
        box_threshold=box_thr,
        text_threshold=text_thr,
        device='cpu'
    )
    src = (image_chw * 255).byte().permute(1,2,0).numpy()
    ann = annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)
    return Image.fromarray(ann), boxes, logits, phrases

def pair_boxes(gts, preds, thr=0.5):
    matched, pairs = set(), []
    for gt in gts:
        best_iou, best_j = thr, None
        g = torch.tensor(gt).unsqueeze(0)
        for j, pr in enumerate(preds):
            if j in matched:
                continue
            p = torch.tensor(pr).unsqueeze(0)
            iou = box_iou(g, p)[0,0].item()
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j is not None:
            pairs.append((gt, preds[best_j]))
            matched.add(best_j)
    return pairs

def iou_box_pairs(pairs):
    global mean_value_boxes
    s, out = 0, []
    for (A, B) in pairs:
        xA, yA = max(A[0],B[0]), max(A[1],B[1])
        xB, yB = min(A[2],B[2]), min(A[3],B[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter <= 0:
            iou = 0.0
        else:
            areaA = abs((A[2]-A[0])*(A[3]-A[1]))
            areaB = abs((B[2]-B[0])*(B[3]-B[1]))
            iou = inter / float(areaA + areaB - inter)
        out.append(iou)
        if iou > 0:
            s += 1
            mean_value_boxes += iou
    if s > 0:
        mean_value_boxes /= s
    return out

def pair_segments(gt_segments, pred_segments, thr=0.5):
    gt_polys = [Polygon(zip(s["x"], s["y"])) for s in gt_segments]
    pr_polys = [Polygon(zip(s["x"], s["y"])) for s in pred_segments]
    scored = []
    for i,g in enumerate(gt_polys):
        for j,p in enumerate(pr_polys):
            if not g.is_valid or not p.is_valid:
                continue
            inter = g.intersection(p).area
            uni   = g.union(p).area
            iou = inter/uni if uni>0 else 0.0
            if iou > 0:
                scored.append((iou,i,j))
    scored.sort(reverse=True, key=lambda t:t[0])
    used_g, used_p, pairs = set(), set(), []
    for iou,i,j in scored:
        if i in used_g or j in used_p:
            continue
        if iou >= thr:
            pairs.append((i,j,iou))
            used_g.add(i); used_p.add(j)
    return pairs

def iou_segments(pairs, gt_segments, pred_segments):
    global mean_value_segments
    s = 0
    ious = []
    for gi, pj, _ in pairs:
        G = Polygon(zip(gt_segments[gi]["x"], gt_segments[gi]["y"]))
        P = Polygon(zip(pred_segments[pj]["x"], pred_segments[pj]["y"]))
        if not G.is_valid or not P.is_valid:
            iou = 0.0
        else:
            inter = G.intersection(P).area
            uni   = G.union(P).area
            iou   = inter/uni if uni>0 else 0.0
        ious.append(iou)
        if iou > 0:
            s += 1
            mean_value_segments += iou
    if s > 0:
        mean_value_segments /= s
    return ious

def calc_prec_recall_f1(n_pred, n_gt, n_tp):
    prec = n_tp / n_pred if n_pred>0 else 0.0
    rec  = n_tp / n_gt   if n_gt>0 else 0.0
    f1   = 0.0 if (prec+rec)==0 else 2*(prec*rec)/(prec+rec)
    return prec, rec, f1

def semantic_iou_2class(gt_segments, pred_segments, H, W):
    # Klasse=1: window, Klasse=0: background
    if H<=0 or W<=0:
        return 0.0
    gt = np.zeros((H,W), dtype=np.uint8)
    pr = np.zeros((H,W), dtype=np.uint8)

    def draw_mask(mask, segs):
        for s in segs:
            pts = np.array(list(zip(s["x"], s["y"])), dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)

    draw_mask(gt, gt_segments)
    draw_mask(pr, pred_segments)

    inter = np.logical_and(gt==1, pr==1).sum()
    union = np.logical_or (gt==1, pr==1).sum()
    if union == 0:
        # keine Fenster in GT und Pred -> mIoU(window)=1.0
        return 1.0
    return float(inter) / float(union)

def to_chw_uint01(img_rgb):
    ten = torch.from_numpy(img_rgb).float()/255.0
    return ten.permute(2,0,1)

def ensure_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def get_interpolation(inter_name):
    if inter_name is None: return None
    if inter_name == "INTER_NEAREST": return cv2.INTER_NEAREST
    if inter_name == "INTER_LINEAR":  return cv2.INTER_LINEAR
    if inter_name == "INTER_CUBIC":   return cv2.INTER_CUBIC
    raise ValueError("Unknown interpolation "+str(inter_name))

# ----------------- Modelle laden -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using cuda' if torch.cuda.is_available() else 'Using cpu')

GROUNDING_DINO_WEIGHTS = "groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG  = "GroundingDINO_SwinT_OGC.py"
grounding_dino_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS).to(device)
print("Grounding DINO model loaded successfully.")

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device)
sam_predictor = SamPredictor(sam)
print("SAM model loaded successfully.")

# ----------------- Bild laden (Originalmaßstab) -----------------
img_org = ensure_rgb(f"{CURRENT_DATASET}.jpg")
org_H, org_W = img_org.shape[:2]
ORG_IMAGE_SIZE = f"{org_W},{org_H}"

# ----------------- Upsample-Variante vorbereiten -----------------
interp = get_interpolation(INTERPOLATION_NAME)

if UPSAMPLING == "RAW":
    work_img = img_org.copy()
    scale = 1.0
    out_H, out_W = org_H, org_W
else:
    # Resize auf quadratische Zielgröße
    work_img = cv2.resize(img_org, (SCALE_UP_SIZE, SCALE_UP_SIZE), interpolation=interp)
    out_H, out_W = SCALE_UP_SIZE, SCALE_UP_SIZE
    # GT-Rescale-Faktor (isotrope Skalierung)
    scale = SCALE_UP_SIZE / float(org_W)

CROPPED_IMAGE_SIZE = f"{out_W},{out_H}"  # kein Crop im Experiment

# ----------------- GroundingDINO -----------------
img_chw = to_chw_uint01(work_img)
_, win_boxes_norm, _, _ = dino(img_chw, VOCAB_SECONDLVL, 0.35, 0.25)

# Optional: zweiter, „großzügigerer“ Versuch (nur wenn erste Suche leer ist)
if len(win_boxes_norm) == 0 and UPSAMPLING != "RAW":
    _, win_boxes_norm, _, _ = dino(img_chw, VOCAB_SECONDLVL, 0.25, 0.20)

# ----------------- Wenn weiterhin keine Detections: Null-Metriken schreiben -----------------
if len(win_boxes_norm) == 0:
    pred_boxes_xyxy = []
    segments_pred = []
    vis = work_img.copy()
    cv2.imwrite(os.path.join(base_dir, f"{DATASET_NUMBER}_{UPSAMPLING}_sam.jpg"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # GT laden & rescalen
    gt_boxes_org = getGTBoxes()
    gt_segs_org  = getGTSegments()
    gt_boxes = rescale_boxes_xyxy(gt_boxes_org, scale)
    gt_segs  = rescale_segments(gt_segs_org, scale)

    # Visuals: GT (grün), keine Preds
    pil = Image.fromarray(work_img.copy()); drw = ImageDraw.Draw(pil)
    for x1,y1,x2,y2 in gt_boxes:
        l,r = sorted([x1,x2]); t,b = sorted([y1,y2])
        drw.rectangle([l,t,r,b], outline="green", width=2)
    pil.save(os.path.join(base_dir, f"{DATASET_NUMBER}_{UPSAMPLING}_gt_pred_boxes.jpg"))

    pil2 = Image.fromarray(work_img.copy()); drw2 = ImageDraw.Draw(pil2)
    for s in gt_segs:
        drw2.polygon(list(zip(s["x"], s["y"])), outline="green")
    pil2.save(os.path.join(base_dir, f"{DATASET_NUMBER}_{UPSAMPLING}_gt_pred_segments.jpg"))

    # Kennzahlen = 0; Semantic mIoU(2c) korrekt berechnen
    gt_box_cnt = len(gt_boxes); pr_box_cnt = 0; tp_box = 0
    prec_box = rec_box = f1_box = 0.0

    gt_seg_cnt = len(gt_segs); pr_seg_cnt = 0; tp_seg = 0
    prec_seg = rec_seg = f1_seg = 0.0

    mean_iou_box = 0.0
    mean_iou_seg = 0.0
    sem_miou_2c  = semantic_iou_2class(gt_segs, [], out_H, out_W)

    print(f"[{UPSAMPLING}] NO DETECTIONS -> writing zeros | Semantic mIoU (2c) = {sem_miou_2c:.3f}")

    # Excel-Export
    TIMEFLAT = time.process_time() - _start_flat
    namespace_excel = {
        "CODE": DATASET_NUMBER,
        "PRECISIONBOX": prec_box,
        "RECALLBOX":    rec_box,
        "F1BOX":        f1_box,
        "PRECISIONSEGMENT": prec_seg,
        "RECALLSEGMENT":    rec_seg,
        "F1SEGMENT":        f1_seg,
        "MEANIOUBOX":       mean_iou_box,
        "MEANIOUSEGMENT":   mean_iou_seg,
        "TIMEFLAT":         TIMEFLAT,
        "TIMEFLATNESTED":   0,
        "TIMENESTED":       0,
        "TYPE": "FLAT",
        "GT_BOX": gt_box_cnt,
        "PRED_BOX": pr_box_cnt,
        "PAIRS_BOX": tp_box,
        "GT_SEGMENT": gt_seg_cnt,
        "PRED_SEGMENT": pr_seg_cnt,
        "PAIRS_SEGMENT": tp_seg,
        "DP2_INDEX": "n/a",
        "VOCAB_GROUNDTRUTH": json.dumps(VOCAB_GROUNDTRUTH),
        "VOCAB_FRSTLVL": "building",  # fix für das Experiment
        "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
        "ORG_IMAGE_SIZE": ORG_IMAGE_SIZE,
        "CROPPED_IMAGE_SIZE": CROPPED_IMAGE_SIZE,
        "SEMANTICIOU":  sem_miou_2c,  # kompatibel
        "SEMANTICMIOU2C": sem_miou_2c,
        "UPSAMPLING": UPSAMPLING
    }

    with open("upsampling_experiment/write_excel_experiment.py", "r", encoding="utf-8") as f:
        exec(f.read(), namespace_excel)

    # sauber beenden – der Distributor fängt SystemExit ab
    raise SystemExit(0)

# ----------------- Ansonsten: mit Preds weitermachen -----------------
# Pred-Boxen in aktuelle Skala bringen
pred_boxes_xyxy = [convertcoords(b, out_W, out_H) for b in win_boxes_norm]

# SAM
sam_predictor.set_image(work_img)
boxes_xyxy_t = torch.tensor(pred_boxes_xyxy, dtype=torch.float32).unsqueeze(0)
boxes_sam = sam_predictor.transform.apply_boxes_torch(boxes_xyxy_t, work_img.shape[:2]).to(device)
masks, _, _ = sam_predictor.predict_torch(
    point_coords=None, point_labels=None, boxes=boxes_sam, multimask_output=False
)

segments_pred = []
vis = work_img.copy()
for m in masks:
    mask_np = m.cpu().numpy().squeeze().astype(np.uint8)
    vis = np.where(mask_np[...,None]>0, np.array([0,255,0],dtype=np.uint8), vis)
    cnts,_ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        xs = [int(p[0][0]) for p in c]
        ys = [int(p[0][1]) for p in c]
        if len(xs) > 2:
            segments_pred.append({"x": xs, "y": ys})

cv2.imwrite(os.path.join(base_dir, f"{DATASET_NUMBER}_{UPSAMPLING}_sam.jpg"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

# GT laden & rescalen
gt_boxes_org = getGTBoxes()
gt_segs_org  = getGTSegments()
gt_boxes = rescale_boxes_xyxy(gt_boxes_org, scale)
gt_segs  = rescale_segments(gt_segs_org, scale)

# Visuals
pil = Image.fromarray(work_img.copy())
drw = ImageDraw.Draw(pil)
for x1,y1,x2,y2 in gt_boxes:
    l,r = sorted([x1,x2]); t,b = sorted([y1,y2])
    drw.rectangle([l,t,r,b], outline="green", width=2)
for x1,y1,x2,y2 in pred_boxes_xyxy:
    l,r = sorted([x1,x2]); t,b = sorted([y1,y2])
    drw.rectangle([l,t,r,b], outline="red", width=2)
pil.save(os.path.join(base_dir, f"{DATASET_NUMBER}_{UPSAMPLING}_gt_pred_boxes.jpg"))

pil2 = Image.fromarray(work_img.copy())
drw2 = ImageDraw.Draw(pil2)
for s in gt_segs:
    drw2.polygon(list(zip(s["x"], s["y"])), outline="green")
for s in segments_pred:
    drw2.polygon(list(zip(s["x"], s["y"])), outline="red")
pil2.save(os.path.join(base_dir, f"{DATASET_NUMBER}_{UPSAMPLING}_gt_pred_segments.jpg"))

# Pairing & IoU
pairs_boxes = pair_boxes(gt_boxes, pred_boxes_xyxy)
_ = iou_box_pairs(pairs_boxes)

pairs_segs  = pair_segments(gt_segs, segments_pred)
_ = iou_segments(pairs_segs, gt_segs, segments_pred)

# Kennzahlen
gt_box_cnt = len(gt_boxes)
pr_box_cnt = len(pred_boxes_xyxy)
tp_box     = len(pairs_boxes)
prec_box, rec_box, f1_box = calc_prec_recall_f1(pr_box_cnt, gt_box_cnt, tp_box)

gt_seg_cnt = len(gt_segs)
pr_seg_cnt = len(segments_pred)
tp_seg     = len(pairs_segs)
prec_seg, rec_seg, f1_seg = calc_prec_recall_f1(pr_seg_cnt, gt_seg_cnt, tp_seg)

mean_iou_box = float(mean_value_boxes)
mean_iou_seg = float(mean_value_segments)

sem_miou_2c  = semantic_iou_2class(gt_segs, segments_pred, out_H, out_W)

print(f"[{UPSAMPLING}] GT_BOX={gt_box_cnt}  PRED_BOX={pr_box_cnt}  TP_BOX={tp_box}")
print(f"[{UPSAMPLING}] Prec/Rec/F1 (Box): {prec_box:.3f}/{rec_box:.3f}/{f1_box:.3f} | mIoU_Box={mean_iou_box:.3f}")
print(f"[{UPSAMPLING}] Prec/Rec/F1 (Seg): {prec_seg:.3f}/{rec_seg:.3f}/{f1_seg:.3f} | mIoU_Seg={mean_iou_seg:.3f}")
print(f"[{UPSAMPLING}] Semantic mIoU (2-class): {sem_miou_2c:.3f}")

# Excel-Export
TIMEFLAT = time.process_time() - _start_flat
namespace_excel = {
    "CODE": DATASET_NUMBER,
    "PRECISIONBOX": prec_box,
    "RECALLBOX":    rec_box,
    "F1BOX":        f1_box,
    "PRECISIONSEGMENT": prec_seg,
    "RECALLSEGMENT":    rec_seg,
    "F1SEGMENT":        f1_seg,
    "MEANIOUBOX":       mean_iou_box,
    "MEANIOUSEGMENT":   mean_iou_seg,
    "TIMEFLAT":         TIMEFLAT,
    "TIMEFLATNESTED":   0,
    "TIMENESTED":       0,
    "TYPE": "FLAT",
    "GT_BOX": gt_box_cnt,
    "PRED_BOX": pr_box_cnt,
    "PAIRS_BOX": tp_box,
    "GT_SEGMENT": gt_seg_cnt,
    "PRED_SEGMENT": pr_seg_cnt,
    "PAIRS_SEGMENT": tp_seg,
    "DP2_INDEX": "n/a",
    "VOCAB_GROUNDTRUTH": json.dumps(VOCAB_GROUNDTRUTH),
    "VOCAB_FRSTLVL": "building",                 # fix im Experiment
    "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
    "ORG_IMAGE_SIZE": ORG_IMAGE_SIZE,
    "CROPPED_IMAGE_SIZE": f"{out_W},{out_H}",
    "SEMANTICIOU":  sem_miou_2c,                 # kompatibel
    "SEMANTICMIOU2C": sem_miou_2c,
    "UPSAMPLING": UPSAMPLING
}

print("ACCESSED")
with open("upsampling_experiment/write_excel_experiment.py", "r", encoding="utf-8") as f:
    exec(f.read(), namespace_excel)
