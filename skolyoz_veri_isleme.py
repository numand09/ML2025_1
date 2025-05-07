"""
Skolyoz Veri İşleme ve Görselleştirme
===================================
Bu kod, omurga röntgen görüntülerinde veri ön işleme ve görselleştirme adımlarını içerir.
Omurga konturlarını tespit etme, önemli noktaları belirleme ve açı ölçümü yapar.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform, measure
import mediapipe as mp
import math


def load_sample_images(data_dir, limit=5):
    """
    Örnek görüntüleri yükler ve görselleştirir
    
    Parametreler:
    data_dir (str): Görüntülerin bulunduğu ana dizin
    limit (int): Her kategoriden kaç görüntü yükleneceği
    """
    categories = ['normal', 'c_curve', 's_curve']
    fig, axes = plt.subplots(len(categories), limit, figsize=(15, 10))
    
    for i, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        try:
            img_files = os.listdir(path)[:limit]
            for j, img_name in enumerate(img_files):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{category}")
                axes[i, j].axis('off')
        except Exception as e:
            print(f"Hata: {path} yüklenemedi - {e}")
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()


def detect_spine_in_xray(image):
    """
    X-ray görüntüsünde omurgayı tespit etme ve görselleştirme
    
    Parametreler:
    image (numpy.ndarray): X-ray görüntüsü
    
    Döndürür:
    processed_image (numpy.ndarray): İşlenmiş görüntü
    spine_curve (numpy.ndarray): Omurga eğrisi noktaları
    """
    # Gri tonlama ve gürültü azaltma
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kontrast artırma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Kenar tespiti
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Kontur bulma
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturları alıp filtreleme (en büyük 5 kontur)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Omurga konturu olabilecek dikey konturları seçme
    spine_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Yükseklik/genişlik oranı yüksek olan konturlar muhtemelen omurga
        if h / w > 2:
            spine_contours.append(cnt)
    
    # Sonuç görüntüsü
    result = image.copy()
    
    # Omurga konturlarını çizme
    cv2.drawContours(result, spine_contours, -1, (0, 255, 0), 2)
    
    # Omurga eğrisini çıkarma
    spine_curve = []
    if spine_contours:
        # Tüm omurga konturlarını birleştir
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in spine_contours])
        
        # Dikey ekseni boyunca sıralama
        sorted_indices = np.argsort(all_points[:, 1])
        sorted_points = all_points[sorted_indices]
        
        # Noktaları yüksekliklerine göre gruplama ve ortalamalarını alma
        y_values = np.unique(sorted_points[:, 1])
        curve_points = []
        
        for y in y_values:
            x_values = sorted_points[sorted_points[:, 1] == y, 0]
            if len(x_values) > 0:
                avg_x = np.mean(x_values)
                curve_points.append([avg_x, y])
        
        spine_curve = np.array(curve_points)
        
        # Eğriyi çizme
        for i in range(1, len(spine_curve)):
            cv2.line(result, 
                    tuple(spine_curve[i-1].astype(int)), 
                    tuple(spine_curve[i].astype(int)), 
                    (255, 0, 0), 2)
    
    return result, spine_curve


def calculate_cobb_angle(spine_curve):
    """
    Omurga eğrisinden Cobb açısını hesaplar
    
    Parametreler:
    spine_curve (numpy.ndarray): Omurga eğrisi noktaları
    
    Döndürür:
    angle (float): Hesaplanan Cobb açısı
    """
    if len(spine_curve) < 10:
        return None
    
    # Eğrinin en üst ve en alt kısmı
    top_curve = spine_curve[:len(spine_curve)//3]
    bottom_curve = spine_curve[2*len(spine_curve)//3:]
    
    # Üst ve alt kısımların eğimi
    if len(top_curve) > 1 and len(bottom_curve) > 1:
        # Üst kısımda lineer regresyon
        x_top = top_curve[:, 0]
        y_top = top_curve[:, 1]
        coeffs_top = np.polyfit(y_top, x_top, 1)
        slope_top = coeffs_top[0]
        
        # Alt kısımda lineer regresyon
        x_bottom = bottom_curve[:, 0]
        y_bottom = bottom_curve[:, 1]
        coeffs_bottom = np.polyfit(y_bottom, x_bottom, 1)
        slope_bottom = coeffs_bottom[0]
        
        # Açıyı hesaplama
        angle_top = math.atan(slope_top) * 180 / math.pi
        angle_bottom = math.atan(slope_bottom) * 180 / math.pi
        
        # Cobb açısı, iki eğimin arasındaki açıdır
        angle = abs(angle_top - angle_bottom)
        
        return angle
    
    return None


def preprocess_image_for_model(image, target_size=(224, 224)):
    """
    Görüntüyü model için hazırlar
    
    Parametreler:
    image (numpy.ndarray): İşlenecek görüntü
    target_size (tuple): Hedef boyut

    Döndürür:
    preprocessed (numpy.ndarray): İşlenmiş görüntü
    """
    # Boyutlandırma
    resized = cv2.resize(image, target_size)
    
    # Normalizasyon
    normalized = resized / 255.0
    
    # Model için boyut genişletme (batch_size için)
    expanded = np.expand_dims(normalized, axis=0)
    
    return expanded


def detect_spine_with_mediapipe(image):
    """
    MediaPipe Pose modeli ile omurga noktalarını tespit eder
    
    Parametreler:
    image (numpy.ndarray): İşlenecek görüntü
    
    Döndürür:
    result_image (numpy.ndarray): İşaretlenmiş görüntü
    spine_points (list): Omurga noktaları
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        # RGB'ye dönüştürme (MediaPipe RGB formatında çalışır)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Sonuç görüntüsü
        result_image = image.copy()
        
        # Omurga noktaları
        spine_points = []
        
        if results.pose_landmarks:
            # Tüm iskelet noktalarını çizme
            mp_drawing.draw_landmarks(
                result_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            # Omurga noktalarını toplama
            landmarks = results.pose_landmarks.landmark
            
            # Omurga ile ilgili noktalar (MediaPipe indeksleri)
            spine_indices = [
                0,   # burun
                11,  # sol omuz
                12,  # sağ omuz
                23,  # sol kalça
                24,  # sağ kalça
            ]
            
            h, w, _ = result_image.shape
            for idx in spine_indices:
                landmark = landmarks[idx]
                # Normalize edilmiş koordinatları piksel koordinatlarına dönüştürme
                x, y = int(landmark.x * w), int(landmark.y * h)
                spine_points.append([x, y])
                
                # Nokta çizme
                cv2.circle(result_image, (x, y), 5, (0, 255, 0), -1)
            
            # Omurga çizgisini çizme
            spine_points = np.array(spine_points)
            for i in range(1, len(spine_points)):
                cv2.line(result_image, 
                         tuple(spine_points[i-1]), 
                         tuple(spine_points[i]), 
                         (255, 0, 0), 2)
    
    return result_image, spine_points


def visualize_data_distribution(labels):
    """
    Veri seti dağılımını görselleştirir
    
    Parametreler:
    labels (numpy.ndarray): Etiketler
    """
    classes = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
    class_counts = np.bincount(labels)
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_counts)
    plt.title('Veri Seti Sınıf Dağılımı')
    plt.xlabel('Sınıf')
    plt.ylabel('Örnek Sayısı')
    
    # Çubukların üzerine sayıları yazma
    for i, count in enumerate(class_counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.savefig('data_distribution.png')
    plt.show()


def visualize_cobb_angles(angles, labels):
    """
    Cobb açılarının dağılımını görselleştirir
    
    Parametreler:
    angles (numpy.ndarray): Cobb açıları
    labels (numpy.ndarray): Etiketler
    """
    plt.figure(figsize=(10, 6))
    
    classes = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
    for i, class_name in enumerate(classes):
        class_angles = angles[labels == i]
        plt.hist(class_angles, bins=10, alpha=0.7, label=class_name)
    
    plt.title('Sınıflara Göre Cobb Açısı Dağılımı')
    plt.xlabel('Cobb Açısı (derece)')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('cobb_angle_distribution.png')
    plt.show()


def process_sample_image(image_path):
    """
    Örnek bir görüntüyü işleyip görselleştirir
    
    Parametreler:
    image_path (str): Görüntü dosyasının yolu
    """
    # Görüntüyü yükleme
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Görüntü yüklenemedi: {e}")
        return
    
    # 1. Orijinal görüntü
    # 2. Omurga tespiti
    # 3. MediaPipe ile tespit
    # 4. İşlenmiş görüntü
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Orijinal görüntü
    axes[0].imshow(image)
    axes[0].set_title("Orijinal Görüntü")
    axes[0].axis('off')
    
    # Omurga tespiti
    spine_image, spine_curve = detect_spine_in_xray(image)
    axes[1].imshow(spine_image)
    axes[1].set_title("Omurga Tespiti")
    axes[1].axis('off')
    
    # MediaPipe ile tespit
    mediapipe_image, spine_points = detect_spine_with_mediapipe(image)
    axes[2].imshow(mediapipe_image)
    axes[2].set_title("MediaPipe Tespiti")
    axes[2].axis('off')
    
    # İşlenmiş görüntü (örneğin kenar tespiti)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    axes[3].imshow(edges_rgb)
    axes[3].set_title("Kenar Tespiti")
    axes[3].axis('off')
    
    # Eğer omurga eğrisi tespit edildiyse Cobb açısını hesapla ve göster
    if spine_curve is not None and len(spine_curve) > 0:
        angle = calculate_cobb_angle(spine_curve)
        if angle is not None:
            plt.suptitle(f"Hesaplanan Cobb Açısı: {angle:.2f} derece", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('processed_sample.png')
    plt.show()