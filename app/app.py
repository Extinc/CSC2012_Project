from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)
def generate():
    camera = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    #Cv2 text initialize
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2

    while True:
        success, image = camera.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:
            for handlandmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, handlandmarks, mpHands.HAND_CONNECTIONS)
                thumbx = handlandmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
                indexx = handlandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x

                # print("thumb: " + thumbx)
                # print("index: " + indexx)
                if thumbx > indexx:
                    image = cv2.putText(image, 'I am a BAKA', org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                    #cv2.putText(image, 'I am a baka.', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)



        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8001, debug=True)
