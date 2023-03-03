from threading import Thread

from flask import Flask, render_template, Response
import cv2
app = Flask(__name__)

camera = cv2.VideoCapture(0)

def capture_frame():
    while True:
        ret, frame = camera.read()
        # Do something with the frame,
        global current_frame
        current_frame = frame

def generate_frame():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    Thread(target=capture_frame).start()
    app.run(debug=True)
