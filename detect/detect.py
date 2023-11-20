import sys
sys.path.insert(0, 'D:/IoT_Project/Parking_Intelligent/yolov7')

from  yolov7.hubconf import custom
from detect.predict import *
from detect.drawbox import *

model_vehicle = custom(path_or_model='D:/IoT_Project/Parking_Intelligent/model/vehicle_detect.pt')
model_plate = custom(path_or_model='D:/IoT_Project/Parking_Intelligent/model/plate_detect.pt')

def take_score(elem):
    return elem[4]

def non_max_suppression(boxes, threshold):
    order = boxes.copy()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            box_i = i.copy()
            box_j = j.copy()
            intersection = max(0, min(box_i[2], box_j[2]) - max(box_i[0], box_j[0])) * \
                           max(0, min(box_i[3], box_j[3]) - max(box_i[1], box_j[1]))
            union = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]) + \
                    (box_j[2] - box_j[0]) * (box_j[3] - box_j[1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep

def detect(frame, plate_content_dict):

    results_vehicle = predict_vehicle(model_vehicle, frame)
    results_plate = predict_plate(model_plate, frame)

    frame = draw_box_vehicle(frame, results_vehicle)
    frame, plate_content_dict = draw_box_plate(frame, results_plate, plate_content_dict)

    return frame, plate_content_dict