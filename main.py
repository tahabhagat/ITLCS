import cv2
import detect
import weighted_count
import numpy as np
import math

#Capturing images
paths = ['images\\1.jpg','images\\2.jpg','images\\3.jpg','images\\4.jpg']
imlst = []
for path in paths:
    imlst.append(cv2.imread(path))

#Loading model

model  = detect.load_model()

#Inferring on each image

result_dict_lst = []
for image in imlst:
    result_dict_lst.append(detect.get_count(image, model))
print(result_dict_lst)

#Getting weighted vehicle count on each image

decayed_weight_lst = []
for result_dict in result_dict_lst:
    decayed_weight_lst.append(weighted_count.weight_decay(result_dict))
print(decayed_weight_lst)

#Checking if the weighted values meet the threshold

decayed_weight_lst = np.array(decayed_weight_lst)
check_threshold = weighted_count.check_threshold(decayed_weight_lst)
print(check_threshold)

max_time = 200
available_time = max_time
time = np.zeros(4)
for i in range(len(check_threshold)):
    if check_threshold[i] == 1:
        time[i] = 60
        available_time -= 60
    elif check_threshold[i] == -1:
        time[i] = 20
        available_time -= 60

sum_of_weights = sum(decayed_weight_lst)
normalised_weights = [i / sum_of_weights for i in decayed_weight_lst]
# print(normalised_weights)

# #Get the time for each side
time = [math.ceil((available_time * normalised_weights[index])+10) if check_threshold[index] == 0 else math.ceil(time[index]) for index in range(len(time))]
print(time)