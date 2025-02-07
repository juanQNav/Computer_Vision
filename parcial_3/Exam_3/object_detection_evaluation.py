import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import plotting as plot
from utils import manage_dataset as md  
from utils import metrics as met

def net_inference(net, image, labels_coco,true_label, threshold=0.5):
    blob = cv2.dnn.blobFromImage(image, swapRB=True)
    net.setInput(blob)

    boxes, _ = net.forward(["detection_out_final", "detection_masks"])

    detection_count = boxes.shape[2]
    detections = []

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = int(box[1])  
        confidence = box[2]

        if confidence > threshold:
            if 0 <= class_id < len(labels_coco):  
                label = labels_coco[class_id]
                x = int(box[3] * image.shape[1]) 
                y = int(box[4] * image.shape[0])
                x2 = int(box[5] * image.shape[1])
                y2 = int(box[6] * image.shape[0])
                
                if label == true_label:
                    detections.append({
                        "label": label,
                        "bbox": (x, y, x2, y2),
                        "confidence": confidence
                    })
            else:
                print(f"Warning: class_id {class_id} is out of range for labels_coco")
    return detections

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",type=str, required=True, help="path to model")
    ap.add_argument("--config",type=str, required=True, help="path to config")
    ap.add_argument("--labels_coco",type=str, required=True, help="path to labels")
    ap.add_argument("--images",type=str, required=True, help="path to images")
    ap.add_argument("--json",type=str, required=False, help="path to json")
    ap.add_argument("--output",type=str, required=False, help="path to output")
    ap.add_argument("--th",type=float, required=False, default=0.5, help="threshold")
    ap.add_argument("--max_imgs",type=int, required=False, default=5, help="max images per class")
    ap.add_argument("--plot", required=False, action="store_true", help="plot the image")

    args = vars(ap.parse_args())

    MODEL_PATH = args["model"]
    CONFIG_PATH = args["config"]
    LABELS_COCO_PATH = args["labels_coco"]
    IMAGE_PATH = args["images"]
    PLOT = args["plot"]
    JSON_PATH = args["json"]
    THRESHOLD = args["th"]
    OUTPUT = args["output"]
    MAX_IMAGES_PER_CLASS = args["max_imgs"]
                                
    os.makedirs(OUTPUT, exist_ok=True)

    selected_labels = ["cat", "dog", "bird", "horse"]

    if JSON_PATH is not None:
        all_images = md.load_data_pascal(JSON_PATH,IMAGE_PATH,selected_labels, max_images_per_class=MAX_IMAGES_PER_CLASS)
    else:
        all_images = md.load_data_custom(IMAGE_PATH,selected_labels)

    net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

    labels = md.load_labels_coco(LABELS_COCO_PATH)

    total_images = sum(len(image_list) for image_list in all_images.values())
    
    detections = {label: [] for label in selected_labels}
    ground_truths = {label: [] for label in selected_labels}

    with tqdm(total=total_images, desc="Inferencing...") as pbar:
        for label, image_list in all_images.items():
            for i, image_info in enumerate(image_list):
                image = image_info["image"]
                true_label = image_info["true_label"]
                true_bbox = image_info["bboxes"]

                ground_truths[true_label].append({
                    "bboxes": true_bbox,
                    "image_id": image_info["image_id"]
                })

                detections_result = net_inference(net, image, labels,true_label ,threshold=THRESHOLD)

                for det in detections_result:
                    if det["label"] in selected_labels:
                        detections[det["label"]].append({
                            "bbox": det["bbox"],
                            "confidence": det["confidence"],
                            "image_id": image_info["image_id"]
                        })
                    else:
                        continue

                res = plot.draw_detections_gtruths(detections_result, [{"bboxes": true_bbox, "label": true_label}], image)
                cv2.imwrite(os.path.join(OUTPUT, f"{true_label}_{i}.png"), res)
                if PLOT:
                    plot.plot_image(res, cmap=None, filename=f"{true_label}_{i}", figsize=None,output= OUTPUT)

                pbar.update(1)

    iou_thresholds = np.arange(0.1, 1.0, 0.05)
    mAP_values = []

    print("Calculating mAP for different IoU thresholds:")
    for iou_thresh in iou_thresholds:
        current_map = met.evaluate_map(detections, ground_truths, iou_threshold=iou_thresh)
        mAP_values.append(current_map)
        print(f"IoU={iou_thresh:.2f}, mAP={current_map:.4f}")

    plot.map_vs_iou(iou_thresholds, mAP_values, output_path=OUTPUT, title='Mean Average Precision (mAP) vs IoU Thresholds')
    plt.show(block=False)
    input("Press Enter to continue...")