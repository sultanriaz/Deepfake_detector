<<<<<<< HEAD
import cv2

img = cv2.imread("D:/downloads/IMG_20230819_174037_667.jpg")


if img is None:
    print("Image not found")
else:
    new=cv2.resize(img, (300,300))
    b,g,r= cv2.split(new)    
    cv2.imshow("new_window", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
=======
import cv2

img = cv2.imread("D:/downloads/IMG_20230819_174037_667.jpg")


if img is None:
    print("Image not found")
else:
    new=cv2.resize(img, (300,300))
    b,g,r= cv2.split(new)    
    cv2.imshow("new_window", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
