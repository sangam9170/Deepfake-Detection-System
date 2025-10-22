# import sys
# from src.exception import CustomException
# from src.logger import logging

# from src.components.data_loader import load_dataset
# from src.components.data_preprocessing import preprocess_images
# from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# from tensorflow.keras.models import load_model


# def evaluate_model(file_path):
#     try:

#         X,y = load_dataset(file_path)


#         X_test, y_test = preprocess_images(X,y)


#         model = load_model('saved_models/model1.h5')

#         y_pred = (model.predict(X_test) > 0.5).astype('int32')

#         print(classification_report(y_test, y_pred))
#         print(confusion_matrix(y_test, y_pred))
#         print(accuracy_score(y_test, y_pred))

#     except Exception as e:
#         raise CustomException(e,sys)

import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_loader import load_dataset
from src.components.data_preprocessing import preprocess_images

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def evaluate_model(file_path):
    try:
        # Load dataset and preprocess
        X, y = load_dataset(file_path)
        X_test, y_test = preprocess_images(X, y)

        # Load trained model
        model = load_model('saved_models/model1.h5')

        # Predict probabilities and classes
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype('int32')

        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

        # -------------------------------
        # ✅ Generate AUC-ROC Curve
        # -------------------------------
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title('ROC Curve for Deepfake Detection')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        logging.info(f"ROC AUC Score: {roc_auc}")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    evaluate_model(r'Dataset\Test')

