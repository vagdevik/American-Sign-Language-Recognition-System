import cv2
import numpy as np

# 0 if laptop cam, 1 if external cam
cap = cv2.VideoCapture(0)

scaling_factorx=1.5
scaling_factory=1.5

while True:
	ret, frame = cap.read() # read and capture frames
	box = cv2.rectangle(frame, (100, 100), (400, 400),(0,255,0), 5)  #green box
	frame=cv2.resize(frame,None,fx=scaling_factorx,fy=scaling_factory,interpolation=cv2.INTER_AREA) #resize window
	frame = cv2.flip( frame, 1 ) #flip image
	cv2.imshow('video output',frame) #display image
	k = cv2.waitKey(10) & 0xff #close if esc key
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()
