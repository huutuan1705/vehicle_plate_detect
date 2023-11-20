import argparse
import cv2
import random
import os
import sys

from deep_sort_realtime.deepsort_tracker import DeepSort
from detect.predict import *

sys.path.insert(0, 'D:/IoT_Project/Parking_Intelligent/yolov7')
from  yolov7.hubconf import custom

tracker = DeepSort()
char_model = custom(path_or_model='D:/IoT_Project/Parking_Intelligent/model/text_recognition.pt')

def draw_box_vehicle(image, boxs):
    for i, (left, top, right, bottom, score, label) in enumerate(boxs):
        color = [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (left, top), (right, bottom)

        cv2.rectangle(image, c1, c2, color= color, thickness= 2) # vex hcn
        cv2.putText(image, "{}".format(label) + " {:.2f}".format(score), (c1[0], c1[1] - 2), 0, fontScale= 0.5, color = color, thickness= 1, lineType= cv2.LINE_AA)
    return image

def draw_box_plate(image, boxs, plate_content_dict):
    tracks = tracker.update_tracks(boxs, frame=image)
    for track in tracks:
        if not track.is_confirmed():
            continue
    
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb

        if track_id not in plate_content_dict:
            plate = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            plate_content = predict_character(char_model, plate)
            plate_content_dict[track_id] = plate_content

        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
        cv2.putText(image, "#" + str(plate_content_dict[track_id]), (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return image, plate_content_dict