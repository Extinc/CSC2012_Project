from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def generate():
    """Generator function to capture video frames and yield them as byte strings."""
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to URL for IP camera
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pass the RGB image through the Holistic model to get body landmark detections
            results = holistic.process(image)

            # Draw the body landmark detections on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks is not None:
                print(f"lEFT ${results.left_hand_landmarks}")

            if results.right_hand_landmarks is not None:
                print(f"right ${results.right_hand_landmarks}")
            # Convert the BGR image back to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a byte string and yield it to the web app
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8001, debug=True)
