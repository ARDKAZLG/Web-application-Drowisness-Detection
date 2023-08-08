from django.shortcuts import render
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
from django.http import HttpResponse, StreamingHttpResponse

video_capture = None

def index(request):
    return render(request, 'index.html')

def start_tracking(request):
    global video_capture
    action = request.POST.get('action')
    if not action:
        if video_capture is None:
            mixer.init()
            sound = mixer.Sound('drowiness_app/resource/alarm.wav')

            face_cascade = cv2.CascadeClassifier('drowiness_app/haar cascade files/haarcascade_frontalface_alt.xml')
            leye_cascade = cv2.CascadeClassifier('drowiness_app/haar cascade files/haarcascade_lefteye_2splits.xml')
            reye_cascade = cv2.CascadeClassifier('drowiness_app/haar cascade files/haarcascade_righteye_2splits.xml')

            lbl = ['Close', 'Open']

            model = load_model('drowiness_app/resource/training_data.h5')
            path = os.getcwd()
            video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            def generate_frames():
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                count = 0
                score = 0
                thicc = 2
                rpred = [99]
                lpred = [99]
                lbl = 'Open'  
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    height, width = frame.shape[:2]

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))

                    left_eyes = leye_cascade.detectMultiScale(gray)
                    right_eyes = reye_cascade.detectMultiScale(gray)

                    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

                    for (x, y, w, h) in faces:
                         if lbl == 'Open':
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle vert pour le visage lorsque les yeux sont ouverts
                         else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rectangle rouge pour le visage lorsque les yeux sont fermés


                    for (x, y, w, h) in right_eyes:
                        r_eye = frame[y:y + h, x:x + w]
                        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color
                        r_eye = cv2.resize(r_eye, (224, 224))
                        r_eye = r_eye / 255.0
                        r_eye = np.expand_dims(r_eye, axis=0)
                        rpred = model.predict(r_eye)
                        if np.argmax(rpred) == 1:
                            lbl = 'Open'
                        else:
                            lbl = 'Closed'
                        break

                    for (x, y, w, h) in left_eyes:
                        l_eye = frame[y:y + h, x:x + w]
                        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color
                        l_eye = cv2.resize(l_eye, (224, 224))
                        l_eye = l_eye / 255.0
                        l_eye = np.expand_dims(l_eye, axis=0)
                        lpred = model.predict(l_eye)
                        if np.argmax(lpred) == 1:
                            lbl = 'Open'
                        else:
                            lbl = 'Closed'
                        break

                    if np.argmax(lpred) == 1 and np.argmax(rpred) == 1:
                        score = 0
                        cv2.putText(frame, "Open", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        score = score + 1
                        cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    if score == 0:
                        sound.stop()
                    elif score > 5:
                        x1,y1,w1,h1 = 0,0,175,75
                        cv2.putText(frame,'Sleep Alert!!',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)    
                        sound.play()
                        if(thicc<16):
                            thicc= thicc+2
                        else:
                            thicc=thicc-2
                            if(thicc<2):
                                thicc=2
                        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 

                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

            return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

    elif action == 'stop':
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            cv2.destroyAllWindows()

    return render(request, 'index.html')
