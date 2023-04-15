import config
import numpy as np
# Get results from detect.py, passed in the function =>vehicle_count

def weight_decay(vehicle_count_dict:dict = {'Car' : 2, 'Motorcycle': 1})->float:

  class_weights = config.class_weights
  #class_weights = {'Car': 1, 'Truck': 1, 'Bus': 1, 'Motorcycle': 1, 'Bicycle' : 1}


  total_decayed_weights = 0


  for class_name, vehicle_count in vehicle_count_dict.items():

    vehicle_count = int(vehicle_count)
    # Weight assigned to a particle vehicle class
    class_weight = class_weights[class_name]
    decayed_class_weight  = 0

    for i in range(vehicle_count):

      # Decayed weight of a vehicle class
      decay_weight_per_vehicle = class_weight * (1 - 1/vehicle_count)**i
      decayed_class_weight += decay_weight_per_vehicle

    # Total decayed weight of all the predicted class
    total_decayed_weights += decayed_class_weight
  return total_decayed_weights 

def check_threshold(decayed_weight_lst):
  check_threshold = np.zeros(4, dtype=int)
  for i in range(len(decayed_weight_lst)):
    if decayed_weight_lst[i] <=config.lower_threshold:
        check_threshold[i] = -1
    elif decayed_weight_lst[i] >=config.upper_threshold:
        check_threshold[i] = 1
    return check_threshold
#print (weight_decay())
# Credits- Atharva Kalbhor
# Copyrights @AK_47