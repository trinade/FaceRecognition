import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image_path = r"C:\Users\Lenovo\PycharmProjects\FaceNet2\iniesta-test2.jpg"
image_path2 = r"C:\Users\Lenovo\Desktop\iniesta-face.jpg"
image_path3 = r"C:\Users\Lenovo\PycharmProjects\FaceNet2\marcelo-test.jpg"
img = cv2.imread(image_path2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    height, width, channels = img.shape
    part_image = img[max(0, y):min(height, h), max(0, x):min(width, w)]
    print(type(part_image))
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
cv2.imshow('img',img)
cv2.waitKey(0)

import tensorflow as tf
import numpy as np



x = tf.constant(np.random.uniform(-1, 1, 10))
y = tf.constant(np.random.uniform(-1, 1, 10))
s = tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0)
print(s)
#image_path = r"C:\Users\Lenovo\PycharmProjects\FaceNet2\cristiano-ronaldo-juventus-2018-19_9pv24viluywd1dqgbynte2tlo.jpg"
#img1 = cv2.imread(image_path, 1)
#newimg = cv2.resize(img1,(96,96))
#cv2.imshow("Hello",img1)
#newimg = cv2.resize(img1,(96,96))
#cv2.imshow("Hello",newimg)
#cv2.waitKey(0)

