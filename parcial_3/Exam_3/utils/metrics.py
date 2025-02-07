import numpy as np 

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def evaluate_map(detections, ground_truths, iou_threshold=0.5):
    aps = []
    for label in ground_truths.keys():
        gt_boxes = ground_truths[label] 
        det_boxes = detections[label]
        
        if not det_boxes or not gt_boxes:
            aps.append(0)
            continue

        det_boxes = sorted(det_boxes, key=lambda x: x["confidence"], reverse=True)
        tp = np.zeros(len(det_boxes))
        fp = np.zeros(len(det_boxes))
        
        matched_gt = set()
        for i, det in enumerate(det_boxes):
            max_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                
                for gt_bbox in gt["bboxes"]:
                    iou = calculate_iou(det["bbox"], gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = j

            if max_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / len(gt_boxes)
        
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    
    return np.mean(aps)