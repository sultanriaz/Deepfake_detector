<<<<<<< HEAD
from mtcnn import MTCNN
import cv2
detector = MTCNN()
img = cv2.imread(r"D:\downloads\download.jpg")


output= detector.detect_faces(img)
if output:

    x , y , width , height = output[0]['box']
    cv2.rectangle(img , (x,y), (x+width, y+height), (0,255,0), 2)
cv2.imshow("pic 1", img)
cv2.waitKey(0)
=======
from mtcnn import MTCNN
import cv2
detector = MTCNN()
img = cv2.imread(r"D:\downloads\download.jpg")


output= detector.detect_faces(img)
if output:

    x , y , width , height = output[0]['box']
    cv2.rectangle(img , (x,y), (x+width, y+height), (0,255,0), 2)
cv2.imshow("pic 1", img)
cv2.waitKey(0)
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
cv2.destroyAllWindows()