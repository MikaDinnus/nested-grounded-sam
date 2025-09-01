from groundingdino.util.inference import load_model, predict, annotate
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
from torchvision.ops import box_iou
import numpy as np
from PIL import Image, ImageDraw
import os
import json
import time
from write_json import save_json
from shapely.geometry import Polygon
from composite_indicator import calcDP2

######### SETUP #########

print("################################" + "STARTING NESTED OPERATION OF " + str(DATASET_NUMBER) + "################################")
CURRENT_DATASET = f"building_facade/ADE_train_{DATASET_NUMBER}"
mean_value_boxes = 0.0
mean_value_segments = 0.0

# start timer for flat + nested
start_flat_nested = time.process_time()

# Bild laden und auf Zielgröße skalieren
image = cv2.cvtColor(cv2.imread(f"{CURRENT_DATASET}.jpg"), cv2.COLOR_BGR2RGB)
ORG_SCALE_SIZE = image.shape[1]  # Breite des Originalbildes
image = cv2.resize(image, (SCALE_UP_SIZE, SCALE_UP_SIZE), interpolation=cv2.INTER_CUBIC)
image = torch.from_numpy(image).float() / 255.0
image = image.permute(2, 0, 1)  # CHW

_, h, w = image.shape
ORG_IMAGE_SIZE = f"{w},{h}"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using cuda' if torch.cuda.is_available() else 'Using cpu')

GROUNDING_DINO_WEIGHTS = "groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG = "GroundingDINO_SwinT_OGC.py"

grounding_dino_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS).to(device)
print("Grounding DINO model loaded successfully.")

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device)
sam_predictor = SamPredictor(sam)
print("SAM model loaded successfully.")

################################

######### EVALUATION FILES SETUP #########

if not os.path.exists(f"evaluation_images/{DATASET_NUMBER}_evaluation"):
    os.makedirs(f"evaluation_images/{DATASET_NUMBER}_evaluation")
if not os.path.exists(f"evaluation_images/{DATASET_NUMBER}_evaluation/nested"):
    os.makedirs(f"evaluation_images/{DATASET_NUMBER}_evaluation/nested")
    print("Created nested evaluation directory for " + DATASET_NUMBER + " dataset.")

################################


######### HELPER FUNCS #########

GT_COUNT_OVERRIDE = None
SEG_GT_COUNT_OVERRIDE = None

# From xc yc w h to x1 y1 x2 y2 with scaled width and height
def convertcoords(boxes, width, height):
    x_center, y_center, w, h = boxes

    x1 = int((x_center - w / 2) * width)
    y1 = int((y_center - h / 2) * height)
    x2 = int((x_center + w / 2) * width)
    y2 = int((y_center + h / 2) * height)

    return x1,y1,x2,y2

# From xc yc w h to x1 y1 x2 y2 without scaling
def xcycwh_to_xyxy(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

# From x1 y1 x2 y2 to xc yc w h
def xyxy_to_xcycwh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

# Returns the area of the box
def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

# DINO Application
def dino(image, caption):
    height, width = image.shape[1], image.shape[2]
    boxes, logits, phrases = predict(
        model=grounding_dino_model,
        image=image,
        caption=caption,
        box_threshold=0.35,
        text_threshold=0.25,
        device='cpu'
    )
    image_source = (image * 255).byte().permute(1, 2, 0).numpy() # Von 0..1 zu 0..255
    print("Image size  in dino (predicting: " + caption + ") after denormalization " + str(image_source.shape))
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return Image.fromarray(annotated_frame), boxes, logits, phrases

# Loads JSON file for given file
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Returns the correct coords
def bbox_from_polygon(x, y):
    return [min(x), min(y), max(x), max(y)]

# Strict find objetcs funcs
def find_objects(data, search_terms, return_polygon = False):
    results = []
    objects = data["annotation"]["object"]
    
    for obj in objects:
        name = obj.get("name", "").lower().strip()
        raw_name = obj.get("raw_name", "").lower().strip()
        found_match = False
        matched_term = None
        
        for term in search_terms:
            term = term.lower().strip()
            
            if term == name or term == raw_name:
                found_match = True
                matched_term = term
                break
            
            name_words = name.split()
            raw_name_words = raw_name.split()
            if term in name_words or term in raw_name_words:
                found_match = True
                matched_term = term
                break
            

            if name.startswith(term) or raw_name.startswith(term):
                if (len(name) == len(term) or 
                    (len(name) > len(term) and name[len(term)] in [' ', '-', '_']) or
                    len(raw_name) == len(term) or 
                    (len(raw_name) > len(term) and raw_name[len(term)] in [' ', '-', '_'])):
                    found_match = True
                    matched_term = term
                    break
        
        if found_match:
            poly = obj.get("polygon", {})
            x = poly.get("x", [])
            y = poly.get("y", [])
            if x and y:
                bbox = bbox_from_polygon(x, y)
                entry = {
                    "id": obj.get("id"),
                    "name": obj.get("name"),
                    "raw_name": obj.get("raw_name"),
                    "bbox": bbox,
                }
                if return_polygon:
                    entry["polygon"] = {"x": x, "y": y}
                results.append(entry)
                print(f"Match für '{matched_term}' -> Name: '{obj.get('name')}', Raw: '{obj.get('raw_name')}'")
    
    print(f"Insg {len(results)} Matches gefunden")
    return results

# Returns the ground truth boxes
def getIoUBboxes():
    filename = f"{CURRENT_DATASET}.json"
    data = load_json(filename)

    search_terms = VOCAB_GROUNDTRUTH

    found = find_objects(data, search_terms)

    output = []

    for obj in found:
         output.append(obj["bbox"])

    return output

# Returns the segmetns of the searched objects ground truth
def getGTSegments():
    filename = f"{CURRENT_DATASET}.json"
    data = load_json(filename)

    search_terms = VOCAB_GROUNDTRUTH

    found = find_objects(data, search_terms, return_polygon=True)

    output = []

    for obj in found:
        output.append(obj["polygon"])

    return output

# Remaps the pred-boxes (prediction made on cropped image) to the original image
def remapbbox(bboxes, crop_x1, crop_y1):
    remapped = []
    for bbox in bboxes:
        xc_crop, yc_crop, w, h = bbox
        xc_orig = xc_crop + crop_x1
        yc_orig = yc_crop + crop_y1
        remapped.append([xc_orig, yc_orig, w, h])
    return remapped

# Rescales the boxes to the new image size (typ. from 256x256 to 512x512)
def rescalebbox(bboxes):
    rescaled =[]
    print("Boxes are scaled up by ", SCALE_UP_VALUE)
    for bbox in bboxes:
        xc, yc, w, h = bbox
        xc_rescaled = xc * SCALE_UP_VALUE
        yc_rescaled = yc * SCALE_UP_VALUE
        w_rescaled = w * SCALE_UP_VALUE
        h_rescaled = h * SCALE_UP_VALUE
        rescaled.append([xc_rescaled, yc_rescaled, w_rescaled, h_rescaled])
    return rescaled

# Rescales the segments to the new image size
def rescalesegment(segments):
    rescaled = []
    print("Segments are scaled up by ", SCALE_UP_VALUE)
    for segment in segments:
        x = segment["x"]
        y = segment["y"]
        x_rescaled = [coord * SCALE_UP_VALUE for coord in x]
        y_rescaled = [coord * SCALE_UP_VALUE for coord in y]
        rescaled.append({"x": x_rescaled, "y": y_rescaled})
    return rescaled

# Converts from normalized coordinates to pixel coordinates
def fromnormtopixel(bboxes, crop_width, crop_height):
    repixeld= []
    for bbox in bboxes:
        xc_norm, yc_norm, w_norm, h_norm = bbox
        xc_pixel = xc_norm * crop_width
        yc_pixel = yc_norm * crop_height
        w_pixel = w_norm * crop_width
        h_pixel = h_norm * crop_height
        repixeld.append([xc_pixel, yc_pixel, w_pixel, h_pixel])
    return repixeld

# Pairs predicted boxes with ground truth boxes (greedy)
def pair_boxes(gts, preds):
    matched_preds = set()
    pairs = []
    iou_threshold = 0.5
    for gt in gts:
        best_iou = iou_threshold
        best_pred = None
        gt_box = torch.tensor(gt).unsqueeze(0)
        for i, pred in enumerate(preds):
            if i in matched_preds:
                continue
            pred_box = torch.tensor(pred).unsqueeze(0)
            iou = box_iou(gt_box, pred_box)[0, 0].item()
            if iou >= best_iou:
                best_iou = iou
                best_pred = i
        if best_pred is not None:
            pairs.append((gt, preds[best_pred]))
            matched_preds.add(best_pred)
    return pairs


# Pairs predicted segments with ground truth segments based on IoU (greedy)
def pair_segments(gt_segments, pred_segments):
    gt_polys = [Polygon(zip(seg['x'], seg['y'])) for seg in gt_segments]
    pred_polys = [Polygon(zip(seg['x'], seg['y'])) for seg in pred_segments]

    iou_threshold = 0.5
    iou_pairs = []
    for i, gt in enumerate(gt_polys):
        for j, pred in enumerate(pred_polys):
            if not gt.is_valid or not pred.is_valid:
                continue
            inter = gt.intersection(pred).area
            union = gt.union(pred).area
            iou = inter / union if union > 0 else 0
            if iou > 0:
                iou_pairs.append((iou, i, j))

    iou_pairs.sort(reverse=True)

    paired_gt = set()
    paired_pred = set()
    pairs = []

    for iou, gt_idx, pred_idx in iou_pairs:
        if gt_idx not in paired_gt and pred_idx not in paired_pred and iou >= iou_threshold:
            pairs.append((gt_idx, pred_idx, iou))
            paired_gt.add(gt_idx)
            paired_pred.add(pred_idx)

    return pairs

# Counts the number of ground truth boxes
def getGTCount():
    filename = f"{CURRENT_DATASET}.json"
    data = load_json(filename)

    search_terms = VOCAB_GROUNDTRUTH

    found = find_objects(data, search_terms)

    output = []

    for obj in found:
        output.append(obj["id"])

    return len(output)


# Calc the IoU and return IoU with boxes and mean value
def iou(pairs):
    length = 0
    global mean_value_boxes
    results = []
    for countID, (boxA, boxB) in enumerate(pairs, start=1):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        if interArea == 0:
            iou_value = 0.0
        else:
            boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
            iou_value = interArea / float(boxAArea + boxBArea - interArea)
            print("After Iteration " + str(length) + " IOU: " + str(iou_value))

        print(f"IoU {countID}: {iou_value:.4f}")
        if iou_value != 0:
            length += 1
            mean_value_boxes = mean_value_boxes + iou_value
            print("Boxes mean-value: " + str(mean_value_boxes)) 
        results.append((iou_value, boxA, boxB))

    if mean_value_boxes != 0:
        mean_value_boxes = mean_value_boxes / length
    print(f"Mean IoU: {mean_value_boxes:.4f}")
    return results

# Calc the IoU and return IoU with segmentID and IoU
def iou_segments(segments_paired, segments_groundtruth, segments_predicted):
    global mean_value_segments
    length = 0
    ious = []
    for gt_idx, pred_idx, iou_val in segments_paired:
        gt_segment = segments_groundtruth[gt_idx]
        pred_segment = segments_predicted[pred_idx]

        gt_points = list(zip(gt_segment["x"], gt_segment["y"]))
        pred_points = list(zip(pred_segment["x"], pred_segment["y"]))

        poly_gt = Polygon(gt_points)
        poly_pred = Polygon(pred_points)

        if not poly_gt.is_valid or not poly_pred.is_valid:
            iou = 0
        else:
            intersection_area = poly_gt.intersection(poly_pred).area
            union_area = poly_gt.union(poly_pred).area
            iou = intersection_area / union_area if union_area != 0 else 0
            print("After Iteration " + str(length) + " IOU: " + str(iou))

        if iou != 0:
            length += 1
            mean_value_segments = mean_value_segments + iou
            print("Segments mean-value: " + str(mean_value_segments))

        print(f"IoU für Segment-Paar ({gt_idx}, {pred_idx}): {iou:.4f}")
        ious.append((iou, gt_segment, pred_segment))

    if mean_value_segments != 0:
        mean_value_segments = mean_value_segments / length

    return ious

# Calc precision value
def calcPrecisionBox():
    pred_count = len(boxes_predicted)
    tp = len(pairs)
    fp = pred_count - tp
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Calc precision value
def calcPrecisionSegment():
    pred_count = len(segments_predicted)
    tp = len(segments_paired)
    fp = pred_count - tp
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Calc recall value
def calcRecallBox():
    gt_count = GT_COUNT_OVERRIDE if GT_COUNT_OVERRIDE is not None else getGTCount()
    tp = len(pairs)
    fn = gt_count - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Calc recall value
def calcRecallSegment():
    gt_count = SEG_GT_COUNT_OVERRIDE if SEG_GT_COUNT_OVERRIDE is not None else len(segments_groundtruth)
    tp = len(segments_paired)
    fn = gt_count - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Calc F1 valuee
def calcF1Box():
    precision = calcPrecisionBox()
    recall = calcRecallBox()
    if (precision + recall) <= 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# Calc F1 valuee
def calcF1Segment():
    precision = calcPrecisionSegment()
    recall = calcRecallSegment()
    if (precision + recall) <= 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def value_to_excel(value):
    return value.item() if hasattr(value, "item") else value

def box_overlaps_crop(b, crop_xyxy):
    x1,y1,x2,y2 = b
    cx1,cy1,cx2,cy2 = crop_xyxy
    ix1, iy1 = max(x1, cx1), max(y1, cy1)
    ix2, iy2 = min(x2, cx2), min(y2, cy2)
    return (ix2 > ix1) and (iy2 > iy1)

def seg_bbox(seg):
    return [min(seg["x"]), min(seg["y"]), max(seg["x"]), max(seg["y"])]

# Helper zur semantic IoU
def polygons_to_mask(polys, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    if not polys:
        return mask
    cv2_polys = []
    for p in polys:
        xs = np.asarray(p["x"], dtype=np.int32)
        ys = np.asarray(p["y"], dtype=np.int32)
        if xs.size < 3 or ys.size < 3:
            continue
        pts = np.stack([xs, ys], axis=1)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        cv2_polys.append(pts.reshape((-1, 1, 2)))
    if cv2_polys:
        cv2.fillPoly(mask, cv2_polys, 1)
    return mask

# s.o
def iou_from_masks_binary(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return np.nan
    return inter / union

# semantic IoU für den Vergleich zu anderen Modellen
def semantic_iou_window_and_miou2c(H, W, gt_segments, pred_segments):
    gt_mask   = polygons_to_mask(gt_segments,   H, W)
    pred_mask = polygons_to_mask(pred_segments, H, W)

    iou_win = iou_from_masks_binary(pred_mask, gt_mask)

    iou_bg = iou_from_masks_binary(1 - pred_mask, 1 - gt_mask)

    vals = [v for v in [iou_win, iou_bg] if not np.isnan(v)]
    miou_2c = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    return (float(iou_win) if not np.isnan(iou_win) else float("nan"),
            miou_2c)

################################

############## DINO FLAT APPLICATION (nur für Gebäudebox) ##################

# Warmup/vis optional
dino(image, VOCAB_FRSTLVL)[0]

################################

############### CROP IMAGE TO FIRST LEVEL VOCAB BBOX #################

image_result, boxes, logits, phrases = dino(image, VOCAB_FRSTLVL)
height, width = image.shape[1], image.shape[2]

# DINO gibt normierte xc,yc,w,h im Bereich [0,1] zurück -> hier nach XYXY in Pixel umrechnen
boxes = [convertcoords(box, width, height) for box in boxes]
areas = [box_area(box) for box in boxes]
max_area_index = areas.index(max(areas))
largest_box = boxes[max_area_index]
x1, y1, x2, y2 = largest_box

# sanftes Padding um die Crop-Box (XYXY beibehalten!)
pad = int(0.02 * max(width, height))
x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
x2 = min(width, x2 + pad); y2 = min(height, y2 + pad)

print("Größte Box:", largest_box)

# Crop anwenden (Tensor CHW)
cropped = image[:, y1:y2, x1:x2]
crop_xyxy = (x1, y1, x2, y2)     # WICHTIG: als XYXY beibehalten
crop_x1, crop_y1 = x1, y1        # für Remapping von XCYCWH (Pixel) -> Global

print(cropped.shape)
_, ch, cw = cropped.shape
CROPPED_IMAGE_SIZE = f"{cw},{ch}"

# Für SAM braucht es HWC uint8
cropped_np = (cropped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

################################

################ DINO NESTED APPLICATION ################

start_nested = time.process_time()
# Fenster/Objekte im Crop detektieren (normierte XCYCWH bezogen auf Crop)
boxes_windows = dino(cropped, VOCAB_SECONDLVL)[1]
image_np_for_sam = cropped_np.copy()

################################

################ SAM APPLICATION ################
if len(boxes_windows) == 0:
    raise RuntimeError("!!INNER CODE ERROR!! NO WINDOWS DETECTED ON " + DATASET_NUMBER + ". DATASET WILL BE SKIPPED")

sam_predictor.set_image(image_np_for_sam)
H, W, _ = image_np_for_sam.shape

# Normierte Crop-Koordinaten -> XYXY in Crop-Pixel
boxes_xyxy = [convertcoords(b, W, H) for b in boxes_windows]
boxes_xyxy_t = torch.tensor(boxes_xyxy, dtype=torch.float32).unsqueeze(0)
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy_t, image_np_for_sam.shape[:2]).to(device)
masks, _, _ = sam_predictor.predict_torch(
    point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False,
)

segments_predicted = []
img_vis = image_np_for_sam.copy()
for mask in masks:
    mask_np = mask.cpu().numpy().squeeze().astype(np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)
    img_vis = np.where(mask_np[..., None] > 0, green, img_vis).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        x = [int(p[0][0]) for p in contour]
        y = [int(p[0][1]) for p in contour]
        if len(x) > 2:
            segments_predicted.append({"x": x, "y": y})

# Segmente vom Crop in globale Koordinaten verschieben
segments_predicted = [
    {"x": [xx + crop_x1 for xx in seg["x"]], "y": [yy + crop_y1 for yy in seg["y"]]}
    for seg in segments_predicted
]

cv2.imwrite(
    f"evaluation_images/{DATASET_NUMBER}_evaluation/nested/{DATASET_NUMBER}_nested_sam.jpg",
    cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
)

################################

############### GET GROUND TRUTH #################

boxes_groundtruth = getIoUBboxes()      # XYXY im Originalmaßstab
dbg = find_objects(load_json(f"{CURRENT_DATASET}.json"), VOCAB_GROUNDTRUTH)
print("GT terms used (name/raw_name):", sorted({(o.get("name","").lower(), o.get("raw_name","").lower()) for o in dbg}))
print("Unique raw_names only:", sorted({o.get("raw_name","").lower() for o in dbg}))
print("VOCAB_GROUNDTRUTH is:", VOCAB_GROUNDTRUTH)
print("GT count (pre-crop):", len(dbg))
segments_groundtruth = getGTSegments()  # Polygone im Originalmaßstab

################################

################ REMAPPING & RESCALING ################

# Normierte Boxen (Crop) -> Pixel im Crop
boxes_windows_pix = fromnormtopixel(boxes_windows, W, H)
# In globale XCYCWH verschieben
boxes_predicted_xcycwh = [[xc + crop_x1, yc + crop_y1, bw, bh] for (xc, yc, bw, bh) in boxes_windows_pix]
# In globale XYXY konvertieren
boxes_predicted = [xcycwh_to_xyxy(b) for b in boxes_predicted_xcycwh]

# GT auf die aktuelle Bildskala (SCALE_UP_SIZE) bringen
SCALE_UP_VALUE = SCALE_UP_SIZE / ORG_SCALE_SIZE
boxes_groundtruth = rescalebbox(boxes_groundtruth)                  # XYXY
segments_groundtruth = rescalesegment(segments_groundtruth)         # Polygone

# GT auf den Crop beschränken (nur einmal!)
boxes_groundtruth = [b for b in boxes_groundtruth if box_overlaps_crop(b, crop_xyxy)]
GT_COUNT_OVERRIDE = len(boxes_groundtruth)

segments_groundtruth = [s for s in segments_groundtruth if box_overlaps_crop(seg_bbox(s), crop_xyxy)]
SEG_GT_COUNT_OVERRIDE = len(segments_groundtruth)

################################

################# DRAW GT & PRED (BOXES) ###############

pilimage = Image.open(f"{CURRENT_DATASET}.jpg").resize((SCALE_UP_SIZE, SCALE_UP_SIZE), Image.LANCZOS)
draw = ImageDraw.Draw(pilimage)

# GT in grün
for box in boxes_groundtruth:
    x0, y0, x1, y1 = box
    left, right = sorted([x0, x1]); top, bottom = sorted([y0, y1])
    draw.rectangle([left, top, right, bottom], outline="green", width=2)

# Pred in rot
for box in boxes_predicted:
    x0, y0, x1, y1 = box
    left, right = sorted([x0, x1]); top, bottom = sorted([y0, y1])
    draw.rectangle([left, top, right, bottom], outline="red", width=2)

pilimage.save(f"evaluation_images/{DATASET_NUMBER}_evaluation/nested/{DATASET_NUMBER}_nested_gt_pred.jpg")

################################

################# DRAW GT & PRED (SEGMENTS) ###############

pilimage = Image.open(f"{CURRENT_DATASET}.jpg").resize((SCALE_UP_SIZE, SCALE_UP_SIZE), Image.LANCZOS)
drawS = ImageDraw.Draw(pilimage)

for segment in segments_groundtruth:
    drawS.polygon(list(zip(segment["x"], segment["y"])), outline="green")
for segment in segments_predicted:
    drawS.polygon(list(zip(segment["x"], segment["y"])), outline="red")

pilimage.save(f"evaluation_images/{DATASET_NUMBER}_evaluation/nested/{DATASET_NUMBER}_nested_gt_pred_segments.jpg")

################################

################ PAIRING & IOU ################

print("GT first (scaled):", boxes_groundtruth[:1])
print("PRED first (global):", boxes_predicted[:1])
print("Same scale? GT~PRED max coord",
      max(map(max, boxes_groundtruth)) if boxes_groundtruth else None,
      max(map(max, boxes_predicted)) if boxes_predicted else None)

pairs = pair_boxes(boxes_groundtruth, boxes_predicted)
print("Paired boxes: ", pairs)

iou_pairs = iou(pairs)

# Visualisierung der Paare über dem Cropbild (optional)
crop_vis = (cropped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
for iou_val, gt_box, pred_box in iou_pairs:
    gx1, gy1, gx2, gy2 = map(int, gt_box)
    px1, py1, px2, py2 = map(int, pred_box)
    cv2.rectangle(crop_vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 1)
    cv2.rectangle(crop_vis, (px1, py1), (px2, py2), (0, 0, 255), 1)

cv2.imwrite(
    f"evaluation_images/{DATASET_NUMBER}_evaluation/nested/{DATASET_NUMBER}_nested_paires_boxes.jpg",
    cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR)
)

segments_paired = pair_segments(segments_groundtruth, segments_predicted)
print("Paired segments: ", segments_paired)
iou_segm = iou_segments(segments_paired, segments_groundtruth, segments_predicted)

pilimage = Image.open(f"{CURRENT_DATASET}.jpg").resize((SCALE_UP_SIZE, SCALE_UP_SIZE), Image.LANCZOS)
drawSE = ImageDraw.Draw(pilimage)
for gt_idx, pred_idx, iou_val in segments_paired:
    gt_seg = segments_groundtruth[gt_idx]
    pred_seg = segments_predicted[pred_idx]
    drawSE.polygon(list(zip(gt_seg["x"], gt_seg["y"])), outline="green")
    drawSE.polygon(list(zip(pred_seg["x"], pred_seg["y"])), outline="red")

pilimage.save(f"evaluation_images/{DATASET_NUMBER}_evaluation/nested/{DATASET_NUMBER}_nested_pairs_segments.jpg")

################################

########### SEMANTIC IoU (pixelbasiert) ###########

H_img = SCALE_UP_SIZE
W_img = SCALE_UP_SIZE

SEMANTIC_IOU, SEMANTIC_MIoU_2C = semantic_iou_window_and_miou2c(
    H_img, W_img, segments_groundtruth, segments_predicted
)

print("Semantic IoU (window, pixelbasiert): ", SEMANTIC_IOU)
print("Semantic mIoU (window + background): ", SEMANTIC_MIoU_2C)

################################

################# METRIKEN AUSGEBEN ################

def calcRecallBox_override():
    gt_count = GT_COUNT_OVERRIDE
    tp = len(pairs)
    fn = max(gt_count - tp, 0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calcRecallSegment_override():
    gt_count = SEG_GT_COUNT_OVERRIDE
    tp = len(segments_paired)
    fn = max(gt_count - tp, 0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calcF1(p, r):
    return 0.0 if (p + r) <= 0 else 2 * (p * r) / (p + r)

print("GT count Box (im Crop): ", GT_COUNT_OVERRIDE)
print("Pred count Box: ", len(boxes_predicted))
print("Pairs Box: ", len(pairs))
prec_box = calcPrecisionBox()
rec_box  = calcRecallBox_override()
print("Precision Box: ", prec_box)
print("Recall Box: ", rec_box)
print("F1 Box: ", calcF1(prec_box, rec_box))
print("Mean IoU of boxes: ", mean_value_boxes)

print("GT count Seg (im Crop): ", SEG_GT_COUNT_OVERRIDE)
print("Pred count Seg: ", len(segments_predicted))
print("Pairs Seg: ", len(segments_paired))
prec_seg = calcPrecisionSegment()
rec_seg  = calcRecallSegment_override()
print("Precision Seg: ", prec_seg)
print("Recall Seg: ", rec_seg)
print("F1 Seg: ", calcF1(prec_seg, rec_seg))
print("Mean IoU of segments: ", mean_value_segments)
print("Original Image Size: ", ORG_IMAGE_SIZE)
print("Cropped Image Size: ", CROPPED_IMAGE_SIZE)

################################

############### WRITE VALIDATION VALUES #################

processtime_flat_nested = time.process_time() - start_flat_nested
processtime_nested = time.process_time() - start_nested

namespace_excel = {
    "CODE": DATASET_NUMBER,
    "PRECISIONBOX": prec_box,
    "RECALLBOX": rec_box,
    "F1BOX": calcF1(prec_box, rec_box),
    "PRECISIONSEGMENT": prec_seg,
    "RECALLSEGMENT": rec_seg,
    "F1SEGMENT": calcF1(prec_seg, rec_seg),
    "MEANIOUBOX": float(mean_value_boxes) if hasattr(mean_value_boxes, "item") else mean_value_boxes,
    "MEANIOUSEGMENT": float(mean_value_segments) if hasattr(mean_value_segments, "item") else mean_value_segments,
    "TIMEFLAT": 0,
    "TIMEFLATNESTED": processtime_flat_nested,
    "TIMENESTED": processtime_nested,
    "TYPE": "NESTED",
    "GT_BOX": GT_COUNT_OVERRIDE,
    "PRED_BOX": len(boxes_predicted),
    "PAIRS_BOX": len(pairs),
    "GT_SEGMENT": SEG_GT_COUNT_OVERRIDE,
    "PRED_SEGMENT": len(segments_predicted),
    "PAIRS_SEGMENT": len(segments_paired),
    "DP2_INDEX": "tba",
    "VOCAB_GROUNDTRUTH": json.dumps(VOCAB_GROUNDTRUTH),
    "VOCAB_FRSTLVL": VOCAB_FRSTLVL,
    "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
    "ORG_IMAGE_SIZE": ORG_IMAGE_SIZE,
    "CROPPED_IMAGE_SIZE": CROPPED_IMAGE_SIZE,
    "SEMANTICIOU": SEMANTIC_IOU,
    "SEMANTICMIOU2C": SEMANTIC_MIoU_2C
}

dp2 = calcDP2(namespace_excel)
namespace_excel["DP2_INDEX"] = dp2

namespace_json = {
    "CODE": DATASET_NUMBER,
    "PRECISIONBOX": prec_box,
    "RECALLBOX": rec_box,
    "F1BOX": calcF1(prec_box, rec_box),
    "PRECISIONSEGMENT": prec_seg,
    "RECALLSEGMENT": rec_seg,
    "F1SEGMENT": calcF1(prec_seg, rec_seg),
    "MEANIOUBOX": namespace_excel["MEANIOUBOX"],
    "MEANIOUSEGMENT": namespace_excel["MEANIOUSEGMENT"],
    "TYPE": "NESTED",
    "GT_BOX": GT_COUNT_OVERRIDE,
    "PRED_BOX": len(boxes_predicted),
    "PAIRS_BOX": len(pairs),
    "GT_SEGMENT": SEG_GT_COUNT_OVERRIDE,
    "PRED_SEGMENT": len(segments_predicted),
    "PAIRS_SEGMENT": len(segments_paired),
    "DP2_INDEX": dp2,
    "VOCAB_GROUNDTRUTH": json.dumps(VOCAB_GROUNDTRUTH),
    "VOCAB_FRSTLVL": VOCAB_FRSTLVL,
    "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
    "ORG_IMAGE_SIZE": ORG_IMAGE_SIZE,
    "CROPPED_IMAGE_SIZE": CROPPED_IMAGE_SIZE,
    "SEMANTICIOU": SEMANTIC_IOU,
    "SEMANTICMIOU2C": SEMANTIC_MIoU_2C
}

with open("write_excel.py") as file:
    exec(file.read(), namespace_excel)

save_json(namespace_json)