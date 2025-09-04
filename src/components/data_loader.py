import cv2
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging


def load_dataset(dir_path, image_size=(128,128)):
    images = []
    labels = []
    classes = ["Real", "Fake"]

    try:
        logging.info("Loading start")
        for label in classes:
            path = os.path.join(dir_path, label)
            for file in os.listdir(path):
                file_path = os.path.join(path,file)
                
                image = cv2.imread(file_path)

                if image is None:
                    continue

                image = cv2.resize(image, image_size)
                images.append(image)

                if label == "Real":
                    labels.append(1)
                else:
                    labels.append(0)
        logging.info("Loading conplete")

        images = np.array(images, dtype='float32') / 255.0
        labels = np.array(labels)

        logging.info("Returning images and labels")
        
        return images, labels
    
    except Exception as e:
        raise CustomException(e,sys)



# if __name__=="__main__":
#     img, labels = load_dataset('Dataset\Train')
#     print(f"image shape : {img.shape}, labels shape : {labels.shape}")





