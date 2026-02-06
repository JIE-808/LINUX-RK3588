import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,640)

count = 500

while True:
    ret,frame = cap.read()
    if ret:
        count+=1
        cv2.imwrite("images/shoushi/10/" + str(count) + '.jpg',frame)
        cv2.imshow("OUTPUT",frame)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        if count >= 550:
            break
cap.release()
cv2.destroyAllWindows()
