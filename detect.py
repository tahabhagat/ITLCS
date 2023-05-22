import cv2
import torch
import get_results
import numpy as np

def load_model(path = 'weights/best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                           #path = 'weights/best.pt'
                           )
    return model

def get_count(img:np.ndarray, model) -> dict:
    results = model(img)
    results.show()
    res_dict = get_results.extract_results(results)
    return res_dict

# path = "night.jpg"
# cv2.imread(path)
# model = load_model()
# get_count(path, model)