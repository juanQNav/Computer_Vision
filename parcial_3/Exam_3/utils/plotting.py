import cv2
import matplotlib.pyplot as plt
import os

def plot_image(image, cmap, filename, figsize, output):
    if image is not None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap=cmap)
        if filename:
            ax.set_title(filename)
        ax.axis('off')
        plt.savefig(os.path.join(output, f"{filename}.png"))
        return fig
    else:
        raise ValueError("No image to display.")

def draw_detections_gtruths(detections, ground_truths, image):
    res = image.copy()
    for gt in ground_truths:
        for bbox in gt['bboxes']:
            x1, y1, x2, y2 = bbox
            label = gt['label']
            res = cv2.rectangle(res, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            cv2.putText(res, f"{label}", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
            cv2.putText(res, f"{label}", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        res = cv2.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(res, f"{label} {confidence:.2f}", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
        cv2.putText(res, f"{label} {confidence:.2f}", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return res

def map_vs_iou(iou_thresholds, mAP_values, figsize=None, output_path=None, title = None):
    plt.figure(figsize=figsize)
    plt.plot(iou_thresholds, mAP_values, marker='o', linestyle='-', color='b', label='mAP')
    plt.title(title)
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP')
    plt.xticks(iou_thresholds, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mAP_vs_IoU.png"))
    
    fig = plt.gcf()
    return fig