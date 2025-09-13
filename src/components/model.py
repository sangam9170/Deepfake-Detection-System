
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from src.exception import CustomException
from src.logger import logging


def build_model(input_shape):
    logging.info("Model building started.")
    try:

        model = Sequential([

            Conv2D(32, (3,3), activation= 'relu', input_shape= input_shape),
            MaxPooling2D(2,2),

            Conv2D(64, (3,3), activation= 'relu'),
            MaxPooling2D(2,2),

            Flatten(),

            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
            

        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        logging.info("Model Building completed.")

    except Exception as e:
        raise CustomException(e,sys)
    
    return model

    