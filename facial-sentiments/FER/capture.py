import cv2
#starting camera
cap=cv2.VideoCapture(0)
#now we can take read input from camera
status,img=cap.read()  #it will take first picture
cv2.imwrite('test_images/test10.jpg',img)
cap.release()
 