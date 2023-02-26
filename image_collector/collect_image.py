import cv2 #opencv
import os
import time
import uuid


IMAGES_PATH = 'model/images'

labels = ['a','b']
number_imgs = 15

for label in labels:
    if not os.path.exists("model/images/"+label):
        os.mkdir("model/images/"+ label)
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format (label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        print('Collecting Image  {}'.format (imgnum))

        imagename = os.path.join(IMAGES_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(1)
        if cv2.waitKey(1) and 0xFF == ord('g'):
            break
    cap.release()