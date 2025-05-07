"""
Skolyoz Tespiti ve Sınıflandırma Modeli
=======================================
Bu kod, omurga röntgen görüntülerinden skolyoz tespiti ve sınıflandırması 
(Normal, C-tipi skolyoz, S-tipi skolyoz) yapmak için bir model oluşturur.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Derin öğrenme kütüphaneleri
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Görüntü işleme kütüphaneleri
import cv2
from skimage import io, transform

# Mediapipe, insan vücudu için temel noktaları tespit etmek için
import mediapipe as mp


# ----- Veri Yükleme ve Ön İşleme -----

def load_image_dataset(data_dir, img_size=(224, 224)):
    """
    Belirtilen dizinden görüntüleri yükler ve ön işlemeden geçirir.
    
    Parametreler:
    data_dir (str): Görüntülerin bulunduğu ana dizin
    img_size (tuple): Görüntülerin yeniden boyutlandırılacağı boyut
    
    Döndürür:
    X (numpy.ndarray): Görüntü verileri
    y (numpy.ndarray): Etiketler (0: Normal, 1: C-tipi skolyoz, 2: S-tipi skolyoz)
    """
    categories = ['normal', 'c_curve', 's_curve']
    X = []
    y = []
    
    for category_idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalizasyon
                
                X.append(img)
                y.append(category_idx)
            except Exception as e:
                print(f"Hata: {img_path} yüklenemedi - {e}")
    
    return np.array(X), np.array(y)


# ----- Görüntü Önişleme Fonksiyonları -----

def extract_spine_contour(image):
    """
    Görüntüden omurga konturu çıkarma işlemi
    """
    # Gri tonlamaya çevirme
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti
    edges = cv2.Canny(blurred, 50, 150)
    
    # Kontur bulma
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturu bulma (muhtemelen omurga)
    max_contour = max(contours, key=cv2.contourArea, default=None)
    
    # Orijinal görüntü üzerine konturu çizme
    result = image.copy()
    if max_contour is not None:
        cv2.drawContours(result, [max_contour], 0, (0, 255, 0), 2)
    
    return result


def detect_spine_keypoints(image):
    """
    MediaPipe ile omurga noktalarını tespit etme
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
        
        # Noktaları çizdirme
        annotated_image = image.copy()
        mp_drawing = mp.solutions.drawing_utils
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
            # Omurga ile ilgili noktaları alma (omuz, kalça, vs.)
            spine_landmarks = []
            # MediaPipe indeksleri: 11-12 (omuzlar), 23-24 (kalçalar)
            spine_indices = [11, 12, 23, 24]
            for idx in spine_indices:
                landmark = results.pose_landmarks.landmark[idx]
                spine_landmarks.append([landmark.x, landmark.y, landmark.z])
            
            return annotated_image, np.array(spine_landmarks)
    
    return annotated_image, None


# ----- CNN Model Oluşturma -----

def create_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Skolyoz sınıflandırması için CNN modeli oluşturur
    
    Parametreler:
    input_shape (tuple): Giriş görüntüsünün boyutu
    num_classes (int): Sınıf sayısı (Normal, C-tipi, S-tipi)
    
    Döndürür:
    model: Eğitilmeye hazır Keras modeli
    """
    model = Sequential([
        # İlk konvolüsyon bloğu
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # İkinci konvolüsyon bloğu
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Üçüncü konvolüsyon bloğu
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Dördüncü konvolüsyon bloğu
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Düzleştirme ve tam bağlantılı katmanlar
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Aşırı öğrenmeyi önlemek için
        Dense(num_classes, activation='softmax')  # Çıkış katmanı
    ])
    
    # Modeli derleme
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ----- Cobb Açısı Regresyon Modeli -----

def create_cobb_angle_model(input_shape=(224, 224, 3)):
    """
    Cobb açısı tahmini için regresyon modeli oluşturur
    
    Parametreler:
    input_shape (tuple): Giriş görüntüsünün boyutu
    
    Döndürür:
    model: Eğitilmeye hazır Keras modeli
    """
    model = Sequential([
        # CNN katmanları (özellik çıkarma)
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Düzleştirme ve regresyon katmanları
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Tek bir değer (Cobb açısı) tahmin etme
    ])
    
    # Regresyon için derleme
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']  # Ortalama mutlak hata
    )
    
    return model


# ----- Örnek Çalıştırma Kodu -----

def main():
    """
    Ana fonksiyon - Tüm iş akışını yönetir
    """
    # Veri yükleme
    print("Veri yükleniyor...")
    # Not: Gerçek bir veri setine sahip olduğunuzda bu kodu kullanabilirsiniz
    # X, y = load_image_dataset("path/to/dataset")
    
    # Şimdilik örnek veri kullanıyoruz
    X = np.random.random((100, 224, 224, 3))  # 100 örnek görüntü
    y = np.random.randint(0, 3, 100)  # Rastgele etiketler
    
    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CNN modeli oluşturma
    print("Model oluşturuluyor...")
    model = create_cnn_model()
    
    # Model özeti
    model.summary()
    
    # Veri artırma
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Erken durdurma ve model kaydetme
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint('best_scoliosis_model.h5', save_best_only=True)
    ]
    
    # Model eğitimi
    print("Model eğitiliyor...")
    # Gerçek verileriniz olduğunda bu kısmı çalıştırabilirsiniz
    """
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=30,
        callbacks=callbacks
    )
    
    # Modeli değerlendirme
    print("Model değerlendiriliyor...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Kaybı: {loss:.4f}")
    print(f"Test Doğruluğu: {accuracy:.4f}")
    
    # Tahminler
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Normal', 'C-Tipi', 'S-Tipi']))
    
    # Karmaşıklık matrisi
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