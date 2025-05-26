from groundingdino.util.inference import load_model, predict, annotate
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
from torchvision.ops import box_iou
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, sys
import json
import time

######### SETUP #########

print("################################" + "STARTING FLAT OPERATION OF " + str(DATASET_NUMBER) + "################################")
CURRENT_DATASET = f"building_facade/ADE_train_0000{DATASET_NUMBER}"
mean_value = 0.0

start_flat = time.process_time()

# Höhe Breite Kanäle
image = cv2.cvtColor(cv2.imread(f"{CURRENT_DATASET}.jpg"), cv2.COLOR_BGR2RGB)
ORG_SCALE_SIZE = image.shape[1]
# Skalieren auf 512x512
image = cv2.resize(image, (SCALE_UP_SIZE, SCALE_UP_SIZE), interpolation=cv2.INTER_LINEAR)
# "In 0-1 Werte
image = torch.from_numpy(image).float() / 255.0
# Kanäle Höhe Breite
image = image.permute(2, 0, 1)

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

######### HELPER FUNCS #########

# From xc yc w h to x1 y1 x2 y2
def convertcoords(boxes, width, height):
    x_center, y_center, w, h = boxes

    x1 = int((x_center - w / 2) * width)
    y1 = int((y_center - h / 2) * height)
    x2 = int((x_center + w / 2) * width)
    y2 = int((y_center + h / 2) * height)

    return x1,y1,x2,y2

previous = None

# From x1 y1 x2 y2 to xc yc w h
def xyxy_to_xcycwh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

# Prepare image for different models
def prepareimage(image, modell, initial):
    global previous
    if previous is not None:
        print("previous preparation was: " + previous)
        if previous == modell:
            print("No new preparation needed")
            return image
    else:
        print("No previous preparation")
    previous = modell

    if initial:
        print("Initial preparation")
        return image

    if modell in ["sam", "groundingdino"]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        image = np.transpose(image, (2, 0, 1))          # HWC to CHW
        image = image.astype(np.float32) / 255.0        # Normalize
    elif modell == "opencv":
        image = np.transpose(image, (1, 2, 0))          # CHW to HWC
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
    else:
        print("Unknown model type. Please use 'sam', 'groundingdino' or 'opencv'.")
        return None
    print("Image prepared for " + modell)
    return image

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
def find_objects(data, search_terms):
    results = []
    objects = data["annotation"]["object"]
    for obj in objects:
        name = obj.get("name", "").lower()
        raw_name = obj.get("raw_name", "").lower()
        for term in search_terms:
            term = term.lower()
            if term in name or term in raw_name:
                poly = obj.get("polygon", {})
                x = poly.get("x", [])
                y = poly.get("y", [])
                if x and y:
                    bbox = bbox_from_polygon(x, y)
                    results.append({
                        "id": obj.get("id"),
                        "name": obj.get("name"),
                        "raw_name": obj.get("raw_name"),
                        "bbox": bbox
                    })
                break
    return results

# Counts the number of ground truth boxes
def getGTCount():
    filename = f"{CURRENT_DATASET}.json"
    data = load_json(filename)

    search_terms = ["window", "windows", "window pane", "pane"]

    found = find_objects(data, search_terms)

    output = []

    for obj in found:
        output.append(obj["id"])

    return len(output)

# Returns the ground truth boxes
def getIoUBboxes():
    filename = f"{CURRENT_DATASET}.json"
    data = load_json(filename)

    search_terms = ["window", "windows", "window pane", "pane"]

    found = find_objects(data, search_terms)

    output = []

    for obj in found:
         output.append(obj["bbox"])

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

# Calc the IoU and return IoU with boxes and mean value
def iou(pairs):
    length = 0
    global mean_value
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
            mean_value = mean_value + iou_value
            print("mean-value: " + str(mean_value)) 
        results.append((iou_value, boxA, boxB))

    if mean_value != 0:
        mean_value = mean_value / length
    print(f"Mean IoU: {mean_value:.4f}")
    return results

# Calc precision value
def calcPrecision():
    pred_count = len(boxes_predicted)
    tp = len(pairs)
    fp = pred_count - tp
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Calc recall value
def calcRecall():
    gt_count = getGTCount()
    tp = len(pairs)
    fn = gt_count - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Calc F1 valuee
def calcF1():
    precision = calcPrecision()
    recall = calcRecall()
    if (precision + recall) <= 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

################################

################ DINO FLAT APPLICATION ################

boxes_windows = dino(image, "windows")[1]
boxes_predicted = [convertcoords(box, image.shape[2], image.shape[1]) for box in boxes_windows]
image = image.permute(1, 2, 0).numpy() # Von CHW zu HWC

################################

################ SAM APPLICATION ################
# when no windows detected stop the code
if len(boxes_windows) == 0:
    raise RuntimeError("!!INNER CODE ERROR!! NO WINDOWS DETECTED ON " + DATASET_NUMBER + ". DATASET WILL BE SKIPPED")
sam_predictor.set_image(image)
# box: normalized box xywh -> unnormalized xyxy
H, W, _ = image.shape
boxes_xyxy = []
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
    mask = mask.cpu().numpy().squeeze()
    image = np.where(mask[..., None], [0, 255, 0], image)

image = image.astype(np.uint8)

cv2.imwrite("output_flat.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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


pilimage.save("boxes_flat.jpg")

################################

################ PAIRING OF GT TO PRED BOXES ################

pairs = pair_boxes(boxes_groundtruth, boxes_predicted)
print("Paired boxes: ", pairs)

################################


################ CALC THE IUO AND DRAW ON IMAGE ################

iou_pairs = iou(pairs)

cropped_np = image

for pair in iou_pairs:
    iou, gt_box, pred_box = pair
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box

    # Draw the boxes on the image
    cv2.rectangle(cropped_np, (int(gt_x1), int(gt_y1)), (int(gt_x2), int(gt_y2)), (255, 0, 0), 1)  # Gt in red
    cv2.rectangle(cropped_np, (int(pred_x1), int(pred_y1)), (int(pred_x2), int(pred_y2)), (0, 0, 255), 1)  # Pred in blue

image = cropped_np.astype(np.uint8)

cv2.imwrite("iou.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

################################


################# CALC PRECISION AND RECALL ###############

print("GT count: ", getGTCount())
print("Pred count: ", len(boxes_predicted))
print("Pairs: ", len(pairs))
print("Precision: ", calcPrecision())
print("Recall: ", calcRecall())
print("F1: ", calcF1())

################################

processtime_flat = time.process_time() - start_flat

namespace= {
    "CODE": DATASET_NUMBER,
    "PRECISION": calcPrecision(),
    "RECALL": calcRecall(),
    "F1": calcF1(),
    "MEANIOU": mean_value,
    "TIMEFLAT": processtime_flat,
    "TIMEFLATNESTED": 0,
    "TIMENESTED": 0,
    "TYPE": "FLAT"
}

with open("write_excel.py") as file:
    exec(file.read(), namespace)