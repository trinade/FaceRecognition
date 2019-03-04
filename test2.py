import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
detector = MTCNN()
#key points is a list of vectors ,each entry list contains points of a particular face
image = cv2.imread("marcelo.jpg")
result = detector.detect_faces(image) #returns a list of dictionaries
print(len(result))
results={}
for i in range(0,len(result)):
    results[i] = result[i]
result_np = np.array(result)
print(result)
print(result_np.shape[0])
