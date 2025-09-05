import cv2 
import numpy as np
import sys

from src.components.data_loader import load_dataset
from src.exception import CustomException
from src.logger import logging




try:
    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
    if face_cascade.empty():
            raise Exception("Error: Haar cascade file not loaded. Check the path.")

    def extract_faces(image, image_size = (128,128)):

        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        

        if len(faces) == 0:
                return cv2.resize(image, image_size)
            
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            return cv2.resize(face, image_size)
            
except Exception as e:
    raise CustomException(e,sys)




def preprocess_images(images_path):
    preprocessed = []

    try:
        logging.info("Preprocess start")
        for img in images_path:
            face = extract_faces(img)

            preprocessed.append(face)
        
        logging.info("Preprocess complete")
        return np.array(preprocessed)/255.0
    
    except Exception as e:
        raise CustomException(e,sys)


if __name__=="__main__":
    img_path, labels = load_dataset(r"Dataset\Train")
    X = preprocess_images(img_path)
    print(X.shape)

