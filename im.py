import cv2

cap = cv2.VideoCapture(0)
print("Camera object created:", cap.isOpened())

cap.release()
