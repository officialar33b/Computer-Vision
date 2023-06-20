import cv2

# Get the smile cascade.\
smile_cascade = cv2.CascadeClassifier("/home/areeb/Desktop/Computer-Vision/smile_cascade.xml")
face_cascade = cv2.CascadeClassifier("/home/areeb/Desktop/Computer-Vision/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('/home/areeb/Desktop/Computer-Vision/haarcascade_eye.xml')


def smile_detector(frame, gray): #frame is the regular image and gray is the blacknwhite image.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #Now we detect smile inside the face frame.
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for sx, sy, sw, sh in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 1)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

    return frame

video = cv2.VideoCapture(0) # cv2.VideoCapture(0) for face cam or you specify the video path.
while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = smile_detector(frame, gray)
    cv2.imshow("Video", detector)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cv2.VideoWriter()
video.release()
cv2.destroyAllWindows()