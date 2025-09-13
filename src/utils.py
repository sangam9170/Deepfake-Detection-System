import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_loader import load_dataset
from src.components.data_preprocessing import preprocess_images
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from tensorflow.keras.models import load_model


def evaluate_model(file_path):
    try:

        X,y = load_dataset(file_path)


        X_test, y_test = preprocess_images(X,y)


        model = load_model('saved_models/model1.h5')

        y_pred = (model.predict(X_test) > 0.5).astype('int32')

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

    except Exception as e:
        raise CustomException(e,sys)



