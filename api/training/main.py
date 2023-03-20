import os
from statistics import mode
from fastapi import FastAPI
import boto3
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
import os
import io
import tempfile

app = FastAPI()
    

@app.get("/")
def train():
    actions = ['a','hello', 'I love you', 'my', 'n', 'name', 'thank', 'y', 'you']
    label_ids = {action: i for i, action in enumerate(actions)}

    s3 =  boto3.client(
                    "s3",
                    endpoint_url=os.environ.get("S3_URL"),
                    aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                    aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
                    )

    bucket_name = os.environ.get('S3_Data_Bucket_Name')
    response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1000)
    
    hand_landmarks = []
    labels = []


    if 'Contents' in response:
        all_object = response['Contents']
        while response.get('NextContinuationToken'):
            response = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=response['NextContinuationToken'], MaxKeys=1000)
            all_object.extend(response['Contents'])
        for obj in all_object:
            if obj['Key'].endswith('label.npy') == False:
                print(obj['Key'])
                path, file = obj['Key'].split('/')
                file_name, extension = os.path.splitext(file)

                response = s3.get_object(Bucket='data', Key=obj['Key'])
                body = response['Body'].read()


                landmark = np.load(io.BytesIO(body))

                if(len(landmark) > 0):
                    hand_landmarks.append(landmark)
                    # print(os.path.join(path,f'{file_name}_label{extension}'))

                    response_label = s3.get_object(Bucket=bucket_name, Key=os.path.join(path,f'{file_name}_label{extension}'))
                    body_label = response_label['Body'].read()
                    label_array = np.load(io.BytesIO(body_label))
                    labels.append(label_array)

        hand_landmarks = np.concatenate(hand_landmarks, axis=0)
        labels = np.concatenate(labels, axis=0)
        for i in range(len(labels)):
            labels[i] = label_ids[labels[i]]
        
        x = np.array(labels)
        labels = x.astype(np.int32)
        labels_one_hot = tf.keras.utils.to_categorical(labels, len(actions))

        train_hand_landmarks, val_hand_landmarks, train_labels, val_labels = train_test_split(
            hand_landmarks, labels_one_hot, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(21,3)))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        early_stopping = EarlyStopping(patience=50, restore_best_weights=True)
        model.fit(train_hand_landmarks, train_labels, epochs=200, validation_data=(val_hand_landmarks, val_labels), callbacks=[early_stopping])

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as temp:
            print("Saving FIle")

            model.save(temp.name, save_format='h5')
            with open(temp.name, 'rb') as temp_h5_file:
                model_buffer = io.BytesIO(temp_h5_file.read())
                model_buffer.seek(0)
                s3.upload_fileobj(model_buffer, "model", "hand_sign_model.h5")
                print("File Uploaded")

        # s3.upload_fileobj(model_bytes, 'model', 'model.h5')
    return {"message": "Hello from Training Services"}