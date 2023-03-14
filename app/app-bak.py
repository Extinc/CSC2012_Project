from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager

app = Flask(__name__)
def generate():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    videos = load_dataset()
    reference_signs = load_reference_signs(videos)
    sign_recorder = SignRecorder(reference_signs)
    webcam_manager = WebcamManager()
    
    #Cv2 text initialize
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, results, sign_detected, is_recording)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break

        cap.release()
        cv2.destroyAllWindows()

    #while True:
        # success, image = camera.read()
        # imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results = hands.process(imageRGB)

        # if results.multi_hand_landmarks:
        #     for handlandmarks in results.multi_hand_landmarks:
        #         mpDraw.draw_landmarks(image, handlandmarks, mpHands.HAND_CONNECTIONS)
        #         thumbx = handlandmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
        #         indexx = handlandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x


        #         if thumbx > indexx:
        #             image = cv2.putText(image, 'I am a BAKA', org, font,
        #                                 fontScale, color, thickness, cv2.LINE_AA)




        # ret, buffer = cv2.imencode('.jpg', image)
        # frame = buffer.tobytes()
        # yield(b'--frame\r\n'
        #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8001, debug=True)
