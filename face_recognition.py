import cv2

# Importing cascades.
face_cascade = cv2.CascadeClassifier('/home/areeb/Desktop/Computer-Vision/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/areeb/Desktop/Computer-Vision/haarcascade_eye.xml')

# Defining a function that does the detection.

# Takes the image input in blacknwhite and returns the output rectangles on the original image.
def detector(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.5, 3) #image, scaling factor(in our case, the image is reduced by 1.3 times), min number of zones.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y,x+w, y+h), (255,0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 55)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey, ex +ew, ey+eh), (0, 255, 0), 2)
    return frame

video_capture = cv2.VideoCapture('uno.mp4')

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detector(gray, frame)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

