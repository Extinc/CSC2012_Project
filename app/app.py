import io
from flask import Flask, render_template, Response, request
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import numpy as np
import os
import tensorflow as tf 
import time
import requests
import tempfile

app = Flask(__name__)

DATA_PATH = os.path.join('data') 
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
recording = None
hand_sign = ''
user_id = ''


actions = np.array(['a','hello', 'I love you', 'my', 'n', 'name', 'thank', 'y', 'you'])
base_url = 'http://0.0.0.0:8080/api/'
model_retrieve = requests.get(f'{base_url}model/retrieve/')

model = None
# Process each frame in the video stream
hand_landmarks_all_frames = []
labels_all_frames = []

def collect_keypoints():
    # create directory if it does not exist
    global count
    """Generator function to capture video frames and yield them as byte strings."""

    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to URL for IP camera
    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pass the RGB image through the Holistic model to get body landmark detections
            results = hands.process(image)



            # Draw landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # print(hand_landmarks.handedness.classification[0].label)
                    global recording    


                    if recording :
                        global hand_landmarks_all_frames, labels_all_frames
                        hand_landmark_tuples = []
                        for landmark in results.multi_hand_landmarks[0].landmark:
                            hand_landmark_tuples.append((landmark.x, landmark.y, landmark.z))
                        
                        hand_landmarks_all_frames.append(hand_landmark_tuples)
                        # hand_landmarks_all_frames.append(hand_landmarks)
                        labels_all_frames.append(hand_sign)

            # Convert the BGR image back to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a byte string and yield it to the web app
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




def predict():

    global count
    """Generator function to capture video frames and yield them as byte strings."""
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to URL for IP camera
    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        sentence = []
        while True:
            ret, frame = cap.read()

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pass the RGB image through the Holistic model to get body landmark detections
            results = hands.process(image)
            
            
            # Draw landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                    hand_landmarks = tf.convert_to_tensor(hand_landmarks, dtype=tf.float32)
                    hand_landmarks = tf.expand_dims(hand_landmarks, axis=0)

                    # Feed the preprocessed hand landmarks to the loaded model to get the predicted class probabilities
                    prediced = model.predict(hand_landmarks)

                    # Get the index of the class with the highest probability to get the predicted gesture label
                    predicted_label = actions[np.argmax(prediced)]
                    if len(sentence) < 1 :
                        sentence.append(predicted_label)
                    if len(sentence) < 5 and len(sentence) > 0:
                        if len(sentence)-1 > -1:
                            if sentence[len(sentence)-1] != predicted_label:
                                sentence.append(predicted_label)
                    else:
                        sentence = [] 
                        sentence.append(predicted_label)
                    # Draw the predicted label on the video frame using OpenCV
                    cv2.putText(image, ' '.join(sentence), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Convert the BGR image back to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a byte string and yield it to the web app
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(collect_keypoints(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record_feed')
def record_feed():
    return Response(predict(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording_route():
    start_recording()
    return ''
@app.route('/stop_recording')
def stop_recording_route():
    stop_recording()
    return ''

output = None

def start_recording():
    global recording, hand_sign,count,user_id
    hand_sign = request.args.get('asl_sign')
    user_id = request.args.get('user_id')
    directory = f'{DATA_PATH}/{hand_sign}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(user_id)
    recording = True

def stop_recording():
    global recording, hand_landmarks_all_frames, hand_sign, labels_all_frames,user_id
    recording = False
    hand_landmarks_all_frames_np = np.array(hand_landmarks_all_frames)
    label_array = np.array(labels_all_frames)
    timestamp = int(time.time())
    np.save(f'{DATA_PATH}/{hand_sign}/{user_id}_{timestamp}', hand_landmarks_all_frames_np)
    hand_landmarks_binary_file = io.BytesIO()   
    label_binary_file = io.BytesIO()   
    np.save(hand_landmarks_binary_file, hand_landmarks_all_frames_np)
    np.save(label_binary_file, label_array)
    requests.post(f'{base_url}numpy/save/{hand_sign}/{user_id}_{timestamp}.npy' , data=hand_landmarks_binary_file.getvalue())
    requests.post(f'{base_url}numpy/save/{hand_sign}/{user_id}_{timestamp}_label.npy' , data=label_binary_file.getvalue())
    hand_landmarks_all_frames.clear()
    labels_all_frames.clear()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def predictpage():
    return render_template('predict.html')



if __name__ == '__main__':
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as temp:
        temp.write(model_retrieve.content)
        temp.flush()
        model = tf.keras.models.load_model(temp.name, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    app.run()
