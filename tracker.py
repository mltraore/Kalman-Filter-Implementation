import cv2
from Detector import detect_inrange
from KalmanFilter import KalmanFilter
import numpy as np

VideoCap=cv2.VideoCapture(0)

KF=KalmanFilter(0.1, [0, 0])

while(True):
    ret, frame=VideoCap.read()

    points, mask=detect_inrange(frame, 800)
    
    etat=KF.predict().astype(np.int32)
    cv2.circle(frame, (int(etat[0]), int(etat[1])), 2, (0, 255, 0), 5)
    cv2.arrowedLine(frame,
                    (int(etat[0]), int(etat[1])), (int(etat[0])+int(etat[2]), int(etat[1])+int(etat[3])),
                    color=(0, 255, 0),
                    thickness=3,
                    tipLength=0.2)
    if (len(points)>0):
        cv2.circle(frame, (points[0][0], points[0][1]), 10, (0, 0, 255), 2)
        KF.update(np.expand_dims(points[0], axis=-1))

    #frame = cv2.resize(frame, (800, 600))
    cv2.imshow('image', frame)
    if mask is not None:
        cv2.imshow('mask', mask)

    if cv2.waitKey(1)&0xFF==ord('q'):
        VideoCap.release()
        cv2.destroyAllWindows()
        break
