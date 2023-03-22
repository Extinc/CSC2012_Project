# %%
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv1D, MaxPooling1D, Flatten, SimpleRNN
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

# %%
DATA_PATH = "data"
actions = ['a', 'c','hello', 'my', 'y', 'n', 'I love you', 'thank', 'you', 'name']
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
            landmark = np.load(os.path.join(sign_path, f'{file_name}{extension}'))
            if(len(landmark) > 0):
                hand_landmarks.append(landmark)
                labels.append(np.load(os.path.join(sign_path,f'{file_name}_label{extension}')))           
        else:
            continue
        
# %% Check 

print(f"{hand_landmarks[95]}")

# %%
hand_landmarks = np.concatenate(hand_landmarks, axis=0)
labels = np.concatenate(labels, axis=0)

# %%
print(f"{hand_landmarks}")


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
print(train_hand_landmarks)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
# %% Define the model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(21, 3)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(len(actions), activation='softmax')
# ])
model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, activation='relu', input_shape=(21,3)))
model.add(SimpleRNN(128, return_sequences=True, activation='relu'))
model.add(SimpleRNN(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# model = Sequential()
# model.add(Conv1D(64, 3, activation='relu', input_shape=(21,3)))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(len(actions), activation='softmax'))

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# %% Confusion Matrix CallBack
from sklearn.datasets import load_iris
import time
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(ConfusionMatrixCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = np.argmax(self.model.predict(x_val), axis=1)
        cm = confusion_matrix(y_val, y_pred)
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        timestamp = int(time.time())
        data = {
            'Actions' : actions,
            'TP' : tp,
            'FP' : fp,
            'FN' : fn,
            'TN' : tn
        }
        df = pd.DataFrame(data)
        df.set_index('Actions')
        df.to_csv('file.csv', mode='a', header=False)
        # np.savetxt(f'epoch_{timestamp}.csv', cm, delimiter=',')



# %% Train the model
# early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
model.fit(train_hand_landmarks, 
          train_labels, epochs=40, 
          validation_data=(val_hand_landmarks, val_labels), 
          callbacks=[tb_callback, ConfusionMatrixCallback(validation_data=(val_hand_landmarks, np.argmax(val_labels, axis=1)))])

# %%
model.summary()


# %%
test_loss, test_accuracy = model.evaluate(val_hand_landmarks, val_labels)

print('Test accuracy:', test_accuracy)

#%%
model.save('hand_gesture_model_CNN2.h5')

# %%
hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
features = model.predict(hand_landmarks.reshape(1, -1))
predicted_label = labels[np.argmax(features)]

# %% Confusion Matrix
confusion_matrix = confusion_matrix(train_labels, predicted_label)

# %%
