import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2
from skimage import io, transform, measure
import mediapipe as mp
import math

def load_sample_images(data_dir, limit=5):
    categories = ['normal', 'c_curve', 's_curve']
    fig, axes = plt.subplots(len(categories), limit, figsize=(15, 10))
    for i, category in enumerate(categories):
        try:
            img_files = os.listdir(os.path.join(data_dir, category))[:limit]
            for j, img_name in enumerate(img_files):
                img = cv2.cvtColor(cv2.imread(os.path.join(data_dir, category, img_name)), cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{category}")
                axes[i, j].axis('off')
        except Exception as e:
            print(f"Hata: {os.path.join(data_dir, category)} yüklenemedi - {e}")
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

def detect_spine_in_xray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(gray, (5, 5), 0))
    edges = cv2.Canny(enhanced, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours = sorted(cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], 
                     key=cv2.contourArea, reverse=True)[:5]
    
    spine_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[3]/cv2.boundingRect(cnt)[2] > 2]
    result = image.copy()
    cv2.drawContours(result, spine_contours, -1, (0, 255, 0), 2)
    
    spine_curve = []
    if spine_contours:
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in spine_contours])
        sorted_points = all_points[np.argsort(all_points[:, 1])]
        
        curve_points = []
        for y in np.unique(sorted_points[:, 1]):
            x_values = sorted_points[sorted_points[:, 1] == y, 0]
            if len(x_values) > A0:
                curve_points.append([np.mean(x_values), y])
        
        spine_curve = np.array(curve_points)
        for i in range(1, len(spine_curve)):
            cv2.line(result, 
                    tuple(spine_curve[i-1].astype(int)), 
                    tuple(spine_curve[i].astype(int)), 
                    (255, 0, 0), 2)
    
    return result, spine_curve

def calculate_cobb_angle(spine_curve):
    if len(spine_curve) < 10:
        return None
    
    top_curve = spine_curve[:len(spine_curve)//3]
    bottom_curve = spine_curve[2*len(spine_curve)//3:]
    
    if len(top_curve) > 1 and len(bottom_curve) > 1:
        coeffs_top = np.polyfit(top_curve[:, 1], top_curve[:, 0], 1)
        coeffs_bottom = np.polyfit(bottom_curve[:, 1], bottom_curve[:, 0], 1)
        
        angle_top = math.atan(coeffs_top[0]) * 180 / math.pi
        angle_bottom = math.atan(coeffs_bottom[0]) * 180 / math.pi
        
        return abs(angle_top - angle_bottom)
    
    return None

def preprocess_image_for_model(image, target_size=(224, 224)):
    return np.expand_dims(cv2.resize(image, target_size) / 255.0, axis=0)

def detect_spine_with_mediapipe(image):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        result_image = image.copy()
        spine_points = []
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(result_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            spine_indices = [0, 11, 12, 23, 24]  # burun, sol omuz, sağ omuz, sol kalça, sağ kalça
            
            h, w, _ = result_image.shape
            for idx in spine_indices:
                landmark = results.pose_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                spine_points.append([x, y])
                cv2.circle(result_image, (x, y), 5, (0, 255, 0), -1)
            
            spine_points = np.array(spine_points)
            for i in range(1, len(spine_points)):
                cv2.line(result_image, tuple(spine_points[i-1]), tuple(spine_points[i]), (255, 0, 0), 2)
    
    return result_image, spine_points

def visualize_data_distribution(labels):
    classes = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
    class_counts = np.bincount(labels)
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_counts)
    plt.title('Veri Seti Sınıf Dağılımı')
    plt.xlabel('Sınıf')
    plt.ylabel('Örnek Sayısı')
    
    for i, count in enumerate(class_counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.savefig('data_distribution.png')
    plt.show()

def visualize_cobb_angles(angles, labels):
    plt.figure(figsize=(10, 6))
    
    classes = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
    for i, class_name in enumerate(classes):
        plt.hist(angles[labels == i], bins=10, alpha=0.7, label=class_name)
    
    plt.title('Sınıflara Göre Cobb Açısı Dağılımı')
    plt.xlabel('Cobb Açısı (derece)')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('cobb_angle_distribution.png')
    plt.show()

def process_sample_image(image_path):
    try:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Görüntü yüklenemedi: {e}")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Orijinal Görüntü")
    axes[0].axis('off')
    
    spine_image, spine_curve = detect_spine_in_xray(image)
    axes[1].imshow(spine_image)
    axes[1].set_title("Omurga Tespiti")
    axes[1].axis('off')
    
    mediapipe_image, spine_points = detect_spine_with_mediapipe(image)
    axes[2].imshow(mediapipe_image)
    axes[2].set_title("MediaPipe Tespiti")
    axes[2].axis('off')
    
    edges_rgb = cv2.cvtColor(cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150), cv2.COLOR_GRAY2RGB)
    axes[3].imshow(edges_rgb)
    axes[3].set_title("Kenar Tespiti")
    axes[3].axis('off')
    
    if spine_curve is not None and len(spine_curve) > 0:
        angle = calculate_cobb_angle(spine_curve)
        if angle is not None:
            plt.suptitle(f"Hesaplanan Cobb Açısı: {angle:.2f} derece", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('processed_sample.png')
    plt.show()