from ultralytics import YOLO

import numpy as np


model = YOLO('./runs/classify/train17/weights/best.pt')  # load a custom model

results = model('/home/202490517/waste-classifier/data/test/plastic/plastic252.jpg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])
print(results[0].boxes)