import cv2
from time import sleep
import torch
import get_results


cam = cv2.VideoCapture(0)
# ret, frame = cam.read()

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'weights/best.pt')
print("\nPress e to infer or q to quit ")

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if cv2.waitKey(25) & 0xFF == ord('e'):
        results = model(frame)
        res_dict = get_results.extract_results(str(results))
        print(res_dict)
        
