import sys
import cv2

from detect.detect import detect 
sys.path.insert(0, 'D:/IoT_Project/Parking_Intelligent/detect')

vid = cv2.VideoCapture(0)
plate_content_dict = {}

count = 0
while(True): 
    count += 1
    if count%8 != 0:
        continue
    ret, frame = vid.read() 

    frame, plate_content_dict = detect(frame, plate_content_dict)
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows() 