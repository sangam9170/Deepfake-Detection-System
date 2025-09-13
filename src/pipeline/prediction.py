import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model('saved_models\\model1.h5')

def prediction(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (128,128))
    image = np.expand_dims(image, axis=0) / 255.0
    pred = model.predict(image)[0][0]
    if pred > 0.5:
        label = "Real"
        confidence = pred
    else:
        label = "Fake"
        confidence = 1 - pred

    return label , round(confidence,2)

if __name__ == "__main__":
    label , confidence = prediction(r"Dataset\real_40803.jpg")
    print(label)
    print(f"Confidence: {confidence}")    



