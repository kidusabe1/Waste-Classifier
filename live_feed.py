from ultralytics import YOLO
import cv2
import time
if __name__ == '__main__':
    model = YOLO("/home/202490517/waste-classifier/runs/classify/train17/weights/best.pt")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        label = names_dict[results[0].top1]
        cv2.putText(frame, label, (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)

        cv2.imshow('detected trash', frame)
        time.sleep(1)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
