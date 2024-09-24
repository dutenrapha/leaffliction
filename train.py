import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        print(class_folder)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                print('File: ', img_path)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))
                aux = np.array(img)
                img_vector = aux.flatten()
                print('img_vector: ', img_vector.shape)
                images.append(img_vector)
                labels.append(label)
                print('Labels: ', label)
                a = np.array(images)
                print(a.shape)
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder('images/teste')
print('Images array: ', images.shape)
images = images.reshape(images.shape[0], -1)  # Flatten images to 2D array

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print('labels_encoded: ', labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
print('y_train: ', y_train, '  y_test: ', y_test)
print('x_train: ', X_train.shape, '  x_test: ', X_test.shape)

model = LGBMClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
