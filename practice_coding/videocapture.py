<<<<<<< HEAD
import cv2
from mtcnn import MTCNN
detector= MTCNN()
cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    output=detector.detect_faces(frame)
    for single_output in output:
        x,y,width,height=single_output['box']
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+width, y+height), color=(255,0,0), thickness=(2)
                      )
    cv2.imshow('win',frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()
=======
import cv2
from mtcnn import MTCNN
detector= MTCNN()
cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    output=detector.detect_faces(frame)
    for single_output in output:
        x,y,width,height=single_output['box']
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+width, y+height), color=(255,0,0), thickness=(2)
                      )
    cv2.imshow('win',frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
