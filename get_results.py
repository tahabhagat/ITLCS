# datatype of result = models.common.Detections

def extract_results(result)->dict:
    """
    Takes as input the model output and returns a dictionary containing the 
    count of classes which are predicted.
    """
    # To get the classes as key and the count of each class as its values
    class_count_dict = {}

    # Generate a list containing only the classes and their count
    result_lst = get_list(result)

    # Even index contains count of class while the odd index contains the name of the class
    for ind in range(0, len(result_lst),2):

        # Remove extra characters from string to get the correct class labels
        class_name = get_labels(result_lst[ind+1])
        
        # Update the dictionary with the count of classes
        class_count_dict[class_name] = result_lst[ind]

    return class_count_dict 

def get_labels(rawClass_name:str)->str:

    """
    Removes commas and converts class names in plural forms to singular form
    Input: String containing the class name in raw form
    Output: String in the form of a class predicted by the model
    """

    # To remove comma from the string
    class_name = rawClass_name[:-1] if rawClass_name[-1] == ',' else rawClass_name

    # To remove the s at the end to get the accurate class name
    if 'Bus' != class_name:
        class_name = class_name[:-1] if class_name[-1] == 's' else class_name


    return class_name

def get_list(raw_result)->list:

    """
        Takes the model result as input and returns a list containing only the class names
        and the count of each class
        Input: image 1/1: 1440x2560 7 space-emptys, 6 space-occupieds
    Speed: 153.1ms pre-process, 3600.8ms inference, 7.8ms NMS per image at shape (1, 3, 1440, 2560)
        Output: ['7', 'space-emptys,', '6', 'space-occupieds']
    """

    result_str = str(raw_result)
    # result_str contains :-
    """image 1/1: 1440x2560 7 space-emptys, 6 space-occupieds
    Speed: 153.1ms pre-process, 3600.8ms inference, 7.8ms NMS per image at shape (1, 3, 1440, 2560)
    """

    result_lst = result_str.split("\n")[0].split(" ")[3:]
    # result_lst contains :-
    # 1. ['7', 'space-emptys,', '6', 'space-occupieds'] OR
    # 2. ['13', 'space-occupieds']                      OR
    # 3. ['13', 'space-emptys']

    return result_lst