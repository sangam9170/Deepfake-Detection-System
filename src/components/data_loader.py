import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging


def load_dataset(dir_path):
    labels = []
    image_paths = []
    classes = ["Real", "Fake"]

    try:
        logging.info("Loading start")
        for label in classes:
            path = os.path.join(dir_path, label)
            for file in os.listdir(path):
                file_path = os.path.join(path,file)
                
                image_paths.append(file_path)

                if label == "Real":
                    labels.append(1)
                else:
                    labels.append(0)
        logging.info("Loading complete")

        labels = np.array(labels)

        logging.info("Returning images and labels")
        
        return image_paths, labels
    
    except Exception as e:
        raise CustomException(e,sys)




