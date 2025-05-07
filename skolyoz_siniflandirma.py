"""
Skolyoz Tespiti ve Sınıflandırma Modeli - Özlü Versiyon
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import cv2
from skimage import io, transform
import mediapipe as mp

def load_image_dataset(data_dir, img_size=(224, 224)):
    categories = ['normal', 'c_curve', 's_curve']
    X, y = [], []
    
    for category_idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img / 255.0
                
                X.append(img)
                y.append(category_idx)
            except Exception as e:
                print(f"Hata: {img_path} yüklenemedi - {e}")
    
    return np.array(X), np.array(y)

def extract_spine_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour = max(contours, key=cv2.contourArea, default=None)
    result = image.copy()
    if max_contour is not None:
        cv2.drawContours(result, [max_contour], 0, (0, 255, 0), 2)
    
    return result

def detect_spine_keypoints(image):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
        
        annotated_image = image.copy()
        mp_drawing = mp.solutions.drawing_utils
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
            spine_landmarks = []
            spine_indices = [11, 12, 23, 24]  # Omuzlar ve kalçalar
            for idx in spine_indices:
                landmark = results.pose_landmarks.landmark[idx]
                spine_landmarks.append([landmark.x, landmark.y, landmark.z])
            
            return annotated_image, np.array(spine_landmarks)
    
    return annotated_image, None

def create_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cobb_angle_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Cobb açısı tahmini
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def main():
    print("Veri yükleniyor...")
    # Gerçek bir veri seti için: X, y = load_image_dataset("path/to/dataset")
    
    # Örnek veri
    X = np.random.random((100, 224, 224, 3))
    y = np.random.randint(0, 3, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Model oluşturuluyor...")
    model = create_cnn_model()
    model.summary()
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint('best_scoliosis_model.h5', save_best_only=True)
    ]
    
    print("Model eğitiliyor...")
    """
    # Gerçek veri ile eğitim
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=30,
        callbacks=callbacks
    )
    
    print("Model değerlendiriliyor...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Kaybı: {loss:.4f}")
    print(f"Test Doğruluğu: {accuracy:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Normal', 'C-Tipi', 'S-Tipi']))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=['Normal', 'C-Tipi', 'S-Tipi'],
              yticklabels=['Normal', 'C-Tipi', 'S-Tipi'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karmaşıklık Matrisi')
    plt.savefig('confusion_matrix.png')
    plt.show()
    """
    
    print("Not: Gerçek veri olmadan eğitim ve değerlendirme işlemleri atlandı.")
    print("Bu script, veri seti hazır olduğunda kullanılabilir.")

if __name__ == "__main__":
    main()