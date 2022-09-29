import cv2
from datetime import datetime, timedelta

vidfile = './20220921_23_35_04.mp4'

cap = cv2.VideoCapture(vidfile)
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps', fps)
starttime = datetime.strptime(vidfile.split('/')[-1].split('.')[0], "%Y%m%d_%H_%M_%S")
print(starttime)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        frame_count+=1
        seconds = round(frame_count / fps)
        video_time = starttime + timedelta(seconds=seconds)
        print(video_time)
        cv2.putText(frame, str(video_time), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)
        cv2.imshow("Image", frame)
        if cv2.waitKey(30) & 0xFF == ord('s'):
            break
        if cv2.waitKey(30) & 0xFF == ord('p'):
            while 1:
                if cv2.waitKey(30) & 0xFF == ord('p'):
                    break
    else:
        break
cap.release()
cv2.destroyAllWindows()
