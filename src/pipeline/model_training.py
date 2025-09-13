import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from src.components.model import build_model
from src.components.data_loader import load_dataset
from src.components.data_preprocessing import preprocess_images


try:

    # Load dataset
    X,y = load_dataset('Dataset\\Train')
    

    #preprocessing
    X_preprossed,y_labels  = preprocess_images(X,y)

    # Splitting dataset in train test
    X_train,X_test,y_train,y_test = train_test_split(X_preprossed,y_labels, test_size=0.2, random_state=42)


    #Build Model
    model = build_model(input_shape=(128,128,3))


    # EarlyStopping
    earlystopping_callbacks = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights= True)


    # Train model

    history = model.fit(
        X_train,y_train,
        validation_data = (X_test, y_test),
        epochs = 20,
        batch_size = 32,
        callbacks = [earlystopping_callbacks]
    )


    logging.info("Training completed.")

    model.save('saved_models/model1.h5')

    logging.info("Model saved.")

except Exception as e:
    raise CustomException(e,sys)


