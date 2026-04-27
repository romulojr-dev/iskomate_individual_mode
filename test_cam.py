import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("Frame captured:", ret)
print("Frame shape:", frame.shape if ret else "None")

cv2.imshow("Test", frame)
cv2.waitKey(3000)  # wait 3 seconds
cv2.destroyAllWindows()
cap.release()