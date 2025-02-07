import json
import os
import cv2

def load_labels_coco(filename):
    with open(filename, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def load_data_pascal(json_file_path, images_dir, target_classes, max_images_per_class=None):
    with open(json_file_path, 'r') as f:
        pascal_data = json.load(f)

    category_mapping = {cat['id']: cat['name'] for cat in pascal_data.get("categories", [])}

    target_category_ids = {cat_id for cat_id, name in category_mapping.items() if name in target_classes}

    images = {label: [] for label in target_classes}

    image_mapping = {img['id']: img for img in pascal_data.get("images", [])}
    
    annotations = pascal_data.get("annotations", [])
    
    image_dict = {}

    for annotation in annotations:
        category_id = annotation['category_id']
        if category_id in target_category_ids:
            category_name = category_mapping[category_id]
            
            image_id = annotation['image_id']
            image_info = image_mapping.get(image_id)
            image_path = os.path.join(images_dir, image_info['file_name'])
            
            image = cv2.imread(image_path)
            
            x, y, w, h = annotation['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            if image_id in image_dict:
                image_dict[image_id]['bboxes'].append((x1, y1, x2, y2))
            else:
                image_dict[image_id] = {
                    'image_id': image_id,
                    'image': image,
                    'bboxes': [(x1, y1, x2, y2)],
                    'true_label': category_name,
                    'file_name': image_info['file_name']
                }

    for image_data in image_dict.values():
        category_name = image_data['true_label']
        if category_name in images:
            if max_images_per_class is None or len(images[category_name]) < max_images_per_class:
                images[category_name].append(image_data)
    return images

def load_data_custom(global_path, labels):
    images = {label: [] for label in labels}

    for label in labels:
        path = os.path.join(global_path, label)
        id = 0
        for image in os.listdir(path):
            id += 1
            image_path = os.path.join(path, image)
            img = cv2.imread(image_path)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[label].append({
                    'image_id': id,
                    'category': label,
                    'image': img
                    })
    return images