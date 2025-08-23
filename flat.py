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
from write_json import save_json
import time
from shapely.geometry import Polygon
from composite_indicator import calcDP2

######### SETUP #########

print("################################" + "STARTING FLAT OPERATION OF " + str(DATASET_NUMBER) + "################################")
CURRENT_DATASET = f"building_facade/ADE_train_{DATASET_NUMBER}"
mean_value_boxes = 0.0
mean_value_segments = 0.0
choose_timer = 0.0

start_flat = time.process_time()

# Höhe Breite Kanäle
image = cv2.cvtColor(cv2.imread(f"{CURRENT_DATASET}.jpg"), cv2.COLOR_BGR2RGB)
ORG_SCALE_SIZE = image.shape[1]
# Skalieren auf 512x512
image = cv2.resize(image, (SCALE_UP_SIZE, SCALE_UP_SIZE), interpolation=cv2.INTER_CUBIC)
# "In 0-1 Werte
image = torch.from_numpy(image).float() / 255.0
# Kanäle Höhe Breite
image = image.permute(2, 0, 1)

_, h, w = image.shape

ORG_IMAGE_SIZE = f"{w},{h}"
CROPPED_IMAGE_SIZE = ORG_IMAGE_SIZE

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
if not os.path.exists(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat"):
    os.makedirs(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat")
    print("Created flat evaluation directory for " + DATASET_NUMBER + " dataset.")

################################

######### HELPER FUNCS #########

# From xc yc w h to x1 y1 x2 y2
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
    print("Image size  in dino after denormalization " + str(image_source.shape))
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

# Looks fpor objects with parameters windows etc. 
def find_objects(data, search_terms, return_polygon = False): # Standardmäßig False, wenn True dann Segmente mitgeben
    results = []
    objects = data["annotation"]["object"] # nimmt alle Objekte aus den Annotations
    for obj in objects:
        name = obj.get("name", "").lower() # nehme den Name (oder "")
        raw_name = obj.get("raw_name", "").lower() # alternativname innerhalb von ade20k
        for term in search_terms:
            term = term.lower()
            if term in name or term in raw_name:
                poly = obj.get("polygon", {}) # extrahiere Segmentierungsppoly 
                x = poly.get("x", [])
                y = poly.get("y", [])
                if x and y:
                    bbox = bbox_from_polygon(x, y) # berechne bbox aus polygon
                    entry = {
                        "id": obj.get("id"),
                        "name": obj.get("name"),
                        "raw_name": obj.get("raw_name"),
                        "bbox": bbox,
                    }
                    if return_polygon: # Wird nur angehangen in der getGTSegments Funktion
                        entry["polygon"] = {"x": x, "y": y}
                    results.append(entry)
                break
    return results

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
    
    for gt in gts:
        best_iou = 0.5
        best_pred = None
        gt_box = torch.tensor(gt).unsqueeze(0)
        
        for i, pred in enumerate(preds):
            if i in matched_preds:
                continue
            pred_box = torch.tensor(pred).unsqueeze(0)
            iou = box_iou(gt_box, pred_box)[0,0].item()
            
            if iou > best_iou:
                best_iou = iou
                best_pred = i
                
        if best_pred is not None:
            pairs.append((gt, preds[best_pred]))
            matched_preds.add(best_pred)
    
    return pairs

# Paitrs predicted segments with ground truth segments based on IoU (greedy)
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
    gt_count = getGTCount()
    tp = len(pairs)
    fn = gt_count - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Calc recall value
def calcRecallSegment():
    gt_count = len(segments_groundtruth)
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

# Func to choose boxes based on the bbox of the nested application to improve the validation possibilities
def chooseboxes(boxes):
    global choose_timer
    print("1) Count of boxes going into triage:", len(boxes))

    if len(boxes) == 0:
        return []
    
    image_result, building_boxes, logits, phrases = dino(image, VOCAB_FRSTLVL)

    if len(building_boxes) == 0:
        return []
    
    width, height = image.shape[2], image.shape[1]
    # nimm nur das größte Gebäude wie im nested Ansatz
    building_box_raw = max(building_boxes, key=lambda box: (box[2]*box[3]))
    building_box_xyxy = convertcoords(building_box_raw, width, height)

    chosen_boxes = []
    start_time = time.process_time()

    # prüfe ob die Boxen innerhalb der Gebäude-Bbox liegen
    for i, box in enumerate(boxes):
        box_xyxy = convertcoords(box, width, height)

        x1, y1, x2, y2 = box_xyxy
        bx1, by1, bx2, by2 = building_box_xyxy

        is_inside = (x1 >= bx1) and (y1 >= by1) and (x2 <= bx2) and (y2 <= by2)

        if is_inside:
            chosen_boxes.append(box)

    end_time = time.process_time()
    choose_timer += (end_time - start_time)
    print("2) Count of boxes surviving triage:", len(chosen_boxes))
    return chosen_boxes

def value_to_excel(value):
    return value.item() if hasattr(value, "item") else value

################################

################ DINO FLAT APPLICATION ################

boxes_dinowindows = dino(image, VOCAB_SECONDLVL)[1]
boxes_windows = chooseboxes(boxes_dinowindows)
boxes_predicted = [convertcoords(box, image.shape[2], image.shape[1]) for box in boxes_windows]
image = image.permute(1, 2, 0).numpy() # Von CHW zu HWC

################################

################ SAM APPLICATION ################
# when no windows detected stop the code only for the given dataset
if len(boxes_windows) == 0:
    raise RuntimeError("!!INNER CODE ERROR!! NO WINDOWS DETECTED ON " + DATASET_NUMBER + ". DATASET WILL BE SKIPPED")
sam_predictor.set_image(image)
# box: normalized box xywh -> unnormalized xyxy
H, W, _ = image.shape
boxes_xyxy = []
segments_predicted = []
for box in boxes_windows:
    boxes_xyxy.append(convertcoords(box, W, H))
boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32).unsqueeze(0)
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(device)
masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
for mask in masks:
    mask_np = mask.cpu().numpy().squeeze().astype(np.uint8)
    image = np.where(mask_np[..., None], [0, 255, 0], image)

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x = [int(p[0][0]) for p in contour]
        y = [int(p[0][1]) for p in contour]

        if len(x) > 2:
            segments_predicted.append({"x": x, "y": y})

image = image.astype(np.uint8)

cv2.imwrite(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat/{DATASET_NUMBER}_flat_sam.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

################################


############### GET GROUND TRUTH BOXES #################

boxes_groundtruth = getIoUBboxes()
print("Groundtruth boxes: ", boxes_groundtruth)
print("Predicted boxes: ", boxes_predicted)

################################


################ RESCALING OF BOXES ################

SCALE_UP_VALUE = SCALE_UP_SIZE / ORG_SCALE_SIZE

print("GT: " + str(rescalebbox(boxes_groundtruth)))
boxes_groundtruth = rescalebbox(boxes_groundtruth)

################################

################# DRAW GROUND TRUTH AND PRED TO IMAGE ###############

pilimage = Image.open(f"{CURRENT_DATASET}.jpg")

pilimage = pilimage.resize((SCALE_UP_SIZE, SCALE_UP_SIZE), Image.LANCZOS)

draw = ImageDraw.Draw(pilimage)

# GT in green
for box in boxes_groundtruth:
    x0, y0, x1, y1 = box
    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    draw.rectangle([left, top, right, bottom], outline="green", width=2)

# Pred in red (rhyme)
for box in boxes_predicted:
    x0, y0, x1, y1 = box
    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    draw.rectangle([left, top, right, bottom], outline="red", width=2)


pilimage.save(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat/{DATASET_NUMBER}_flat_gt_pred.jpg")

################################


################# DRAW GROUND TRUTH AND PRED TO IMAGE (SEGMENTS) ###############

pilimage = Image.open(f"{CURRENT_DATASET}.jpg")
pilimage = pilimage.resize((SCALE_UP_SIZE, SCALE_UP_SIZE), Image.LANCZOS)
drawS = ImageDraw.Draw(pilimage)

segments_groundtruth = rescalesegment(getGTSegments())

# GT in grün
for segment in segments_groundtruth:
    points = list(zip(segment["x"], segment["y"]))
    drawS.polygon(points, outline="green")

# Pred in rot
for segment in segments_predicted:
    points = list(zip(segment["x"], segment["y"]))
    drawS.polygon(points, outline="red")


pilimage.save(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat/{DATASET_NUMBER}_flat_gt_pred_segments.jpg")

################################


################ PAIRING OF GT TO PRED BOXES ################

pairs = pair_boxes(boxes_groundtruth, boxes_predicted)
print("Paired boxes: ", pairs)

################################

################ CALC THE IOU AND DRAW ON IMAGE (BOXES) ################

iou_pairs = iou(pairs)

# Convert cropped tensor to a NumPy array
cropped_np = image

for pair in iou_pairs:
    iou, gt_box, pred_box = pair
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box

    # Draw the boxes on the image
    cv2.rectangle(cropped_np, (int(gt_x1), int(gt_y1)), (int(gt_x2), int(gt_y2)), (255, 0, 0), 1)  # Gt in red
    cv2.rectangle(cropped_np, (int(pred_x1), int(pred_y1)), (int(pred_x2), int(pred_y2)), (0, 0, 255), 1)  # Pred in blue

image = cropped_np.astype(np.uint8)

cv2.imwrite(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat/{DATASET_NUMBER}_flat_paires_boxes.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

################################


################ PAIRING OF GT TO PRED SEGMENTS ################

segments_paired = pair_segments(segments_groundtruth, segments_predicted)
print("Paired segments: ", segments_paired)

################################


################ CALC THE IOU AND DRAW PAIRS ON IMAGE (SEGMENTS) ################

iou_segm = iou_segments(segments_paired, segments_groundtruth, segments_predicted)

pilimage = Image.open(f"{CURRENT_DATASET}.jpg")
pilimage = pilimage.resize((SCALE_UP_SIZE, SCALE_UP_SIZE), Image.LANCZOS)
drawSE = ImageDraw.Draw(pilimage)

for pair in segments_paired:
        gt_idx, pred_idx, iou = pair
        gt_segment = segments_groundtruth[gt_idx]
        gt_points = list(zip(gt_segment["x"], gt_segment["y"]))

        pred_segment = segments_predicted[pred_idx]
        pred_points = list(zip(pred_segment["x"], pred_segment["y"]))

        drawSE.polygon(gt_points, outline="green")
        drawSE.polygon(pred_points, outline="red")

pilimage.save(f"evaluation_images/{DATASET_NUMBER}_evaluation/flat/{DATASET_NUMBER}_flat_pairs_segments.jpg")

################################


################# CALC PRECISION AND RECALL ###############

print("GT count Box: ", getGTCount())
print("Pred count Box: ", len(boxes_predicted))
print("Pairs Box: ", len(pairs))
print("Precision Box: ", calcPrecisionBox())
print("Recall Box: ", calcRecallBox())
print("F1 Box: ", calcF1Box())
print("Mean IoU of boxes: ", mean_value_boxes)
print("GT count Seg: ", len(segments_groundtruth))
print("Pred count Seg: ", len(segments_predicted))
print("Pairs Seg: ", len(segments_paired))
print("Precision Seg: ", calcPrecisionSegment())
print("Recall Seg: ", calcRecallSegment())
print("F1 Seg: ", calcF1Segment())
print("Mean IoU of segments: ", mean_value_segments)


################################

processtime_flat = time.process_time() - start_flat - choose_timer

namespace_excel= {
    "CODE": DATASET_NUMBER,
    "PRECISIONBOX": value_to_excel(calcPrecisionBox()),
    "RECALLBOX": value_to_excel(calcRecallBox()),
    "F1BOX": value_to_excel(calcF1Box()),
    "PRECISIONSEGMENT": value_to_excel(calcPrecisionSegment()),
    "RECALLSEGMENT": value_to_excel(calcRecallSegment()),
    "F1SEGMENT": value_to_excel(calcF1Segment()),
    "MEANIOUBOX": value_to_excel(mean_value_boxes),
    "MEANIOUSEGMENT": value_to_excel(mean_value_segments),
    "TIMEFLAT": value_to_excel(processtime_flat),
    "TIMEFLATNESTED": 0,
    "TIMENESTED": 0,
    "TYPE": "FLAT",
    "GT_BOX": getGTCount(),
    "PRED_BOX": len(boxes_predicted),
    "PAIRS_BOX": len(pairs),
    "GT_SEGMENT": len(segments_groundtruth),
    "PRED_SEGMENT": len(segments_predicted),
    "PAIRS_SEGMENT": len(segments_paired),
    "DP2_INDEX": "tba",
    "VOCAB_GROUNDTRUTH": VOCAB_GROUNDTRUTH,
    "VOCAB_FRSTLVL": VOCAB_FRSTLVL,
    "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
    "ORG_IMAGE_SIZE": ORG_IMAGE_SIZE,
    "CROPPED_IMAGE_SIZE": CROPPED_IMAGE_SIZE
}

dp2 = calcDP2(namespace_excel)

namespace_excel["DP2_INDEX"] = value_to_excel(dp2)

namespace_json = {
    "CODE": DATASET_NUMBER,
    "PRECISIONBOX": value_to_excel(calcPrecisionBox()),
    "RECALLBOX": value_to_excel(calcRecallBox()),
    "F1BOX": value_to_excel(calcF1Box()),
    "PRECISIONSEGMENT": value_to_excel(calcPrecisionSegment()),
    "RECALLSEGMENT": value_to_excel(calcRecallSegment()),
    "F1SEGMENT": value_to_excel(calcF1Segment()),
    "MEANIOUBOX": value_to_excel(mean_value_boxes),
    "MEANIOUSEGMENT": value_to_excel(mean_value_segments),
    "TYPE": "FLAT",
    "GT_BOX": getGTCount(),
    "PRED_BOX": len(boxes_predicted),
    "PAIRS_BOX": len(pairs),
    "GT_SEGMENT": len(segments_groundtruth),
    "PRED_SEGMENT": len(segments_predicted),
    "PAIRS_SEGMENT": len(segments_paired),
    "DP2_INDEX": value_to_excel(dp2),
    "VOCAB_GROUNDTRUTH": VOCAB_GROUNDTRUTH,
    "VOCAB_FRSTLVL": VOCAB_FRSTLVL,
    "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
    "ORG_IMAGE_SIZE": ORG_IMAGE_SIZE,
    "CROPPED_IMAGE_SIZE": CROPPED_IMAGE_SIZE
}

with open("write_excel.py") as file:
    exec(file.read(), namespace_excel)

save_json(namespace_json)