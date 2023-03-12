# %%
import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# %%
DATA_PATH = "data"
actions = ['a','b', 'c']
label_ids = {action: i for i, action in enumerate(actions)}

# %%
hand_landmarks = []
labels = []

for asl_sign in actions:
    sign_path = os.path.join(DATA_PATH, asl_sign)
    sign_files = os.listdir(sign_path)
    sign_files.sort()
    
    for file in sign_files:
        file_name, extension = os.path.splitext(file)
        if (file_name.endswith("label") == False):
            hand_landmarks.append(np.load(os.path.join(sign_path, f'{file_name}{extension}')))
            labels.append(np.load(os.path.join(sign_path,f'{file_name}_label{extension}' )))           
        else:
            continue
        
# %% Check 

print(f"{labels}")

# %%
hand_landmarks = np.concatenate(hand_landmarks, axis=0)
labels = np.concatenate(labels, axis=0)


#%%
for i in range(len(labels)):
    labels[i] = label_ids[labels[i]]

#%%  CHECK Label

x = np.array(labels)
labels = x.astype(np.int32)
print(hand_landmarks)


# %%

labels_one_hot = tf.keras.utils.to_categorical(labels, len(actions))

#%% 
print(labels_one_hot)

# %%
train_hand_landmarks, val_hand_landmarks, train_labels, val_labels = train_test_split(
    hand_landmarks, labels_one_hot, test_size=0.2, random_state=42)

# %%
print(train_labels)


# %% Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(21, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation='softmax')
])


# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% Train the model
model.fit(train_hand_landmarks, train_labels, epochs=10, validation_data=(val_hand_landmarks, val_labels))

# %%
model.summary()


# %%
test_loss, test_accuracy = model.evaluate(val_hand_landmarks, val_labels)

print('Test accuracy:', test_accuracy)

#%%
model.save('hand_gesture_model.h5')

# %%
hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
features = model.predict(hand_landmarks.reshape(1, -1))
predicted_label = labels[np.argmax(features)]