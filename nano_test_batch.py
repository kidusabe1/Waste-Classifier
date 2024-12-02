from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import cv2
import os
import torch
from ultralytics import YOLO

import numpy as np

ground_truth = {}
classes = {'batteries': 0, 'clothes': 1, 'glass': 2, 'metal': 3, 'organic': 4, 'paper': 5, 'plastic': 6}
model = YOLO("./runs/classify/train17/weights/best.pt")  # Replace with your model path

# Path to your custom test dataset
test_dataset_path = "/home/202490517/waste-classifier/data/test/"
for class_dir in os.listdir(test_dataset_path):
    class_path = os.path.join(test_dataset_path, class_dir)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        ground_truth[img_name] = classes[class_dir]
y_pred = []
y_true = []

for class_dir in os.listdir(test_dataset_path):
    class_path = os.path.join(test_dataset_path, class_dir)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        # img = cv2.imread(image_path)
        # if img is None:
        #     print(f"Image {img_name} not found.")
        #     continue

        results = model(img_path)
        predicted_label = results[0].probs.top1
        y_true.append(ground_truth[img_name])
        y_pred.append(predicted_label)