import cv2
import numpy as np
import mediapipe as mp
import math
from SkolyozFramework import BaseImageProcessor

class SkolyozImageProcessor(BaseImageProcessor):
    def __init__(self, img_size=(224, 224)):
        super().__init__(img_size)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def preprocess(self, image):
        return np.expand_dims(self.normalize_image(self.resize_image(image)), axis=0)

    def detect_spine_in_xray(self, image):
        """X-ray görüntüsünde omurgayı tespit eden geliştirilmiş method"""
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Görüntü iyileştirme
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Kenar tespiti ve morfolojik işlemler
        edges = cv2.Canny(blurred, 30, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Dikey yapıyı vurgulamak için morfolojik işlem
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_edges = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Omurga merkez çizgisini bulmak için görüntüyü dikey olarak incelt
        h, w = vertical_edges.shape
        center_strip = vertical_edges[:, w//3:2*w//3]
        
        # Omurga noktalarını bul
        spine_points = []
        step = h // 20  # 20 nokta al
        
        for y in range(0, h, step):
            if y + step > h:
                continue
            strip = center_strip[y:y+step, :]
            if np.sum(strip) > 0:  # Bu kesitte omurga var mı?
                white_pixels = np.where(strip > 0)
                if len(white_pixels[0]) > 0:
                    avg_x = int(np.mean(white_pixels[1])) + w//3  # x koordinatını orijinal görüntüye ayarla
                    avg_y = y + step//2
                    spine_points.append([avg_x, avg_y])
        
        # Omurga noktaları arasında interpolasyon yap
        if len(spine_points) >= 2:
            # Yumuşak bir eğri elde etmek için interpolasyon
            spine_points = np.array(spine_points)
            # Y'ye göre sırala
            spine_points = spine_points[spine_points[:, 1].argsort()]
            
            # Az sayıda nokta varsa daha fazla nokta ekle
            if len(spine_points) < 10:
                y_values = np.linspace(spine_points[0][1], spine_points[-1][1], 20)
                x_interp = np.interp(y_values, spine_points[:, 1], spine_points[:, 0])
                spine_points = np.column_stack((x_interp, y_values))
        else:
            # Omurga tespit edilemezse MediaPipe ile dene
            return image.copy(), []
        
        # Sonuç görüntüsünü hazırla
        result = image.copy()
        
        # Omurga noktalarını ve eğriyi çiz
        for point in spine_points:
            cv2.circle(result, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
        
        # Omurga eğrisini çiz
        for i in range(1, len(spine_points)):
            p1 = (int(spine_points[i-1][0]), int(spine_points[i-1][1]))
            p2 = (int(spine_points[i][0]), int(spine_points[i][1]))
            cv2.line(result, p1, p2, (255, 0, 0), 2)
        
        return result, spine_points

    def detect_spine_with_mediapipe(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        with self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            result = pose.process(img_rgb)
            output = image.copy()
            points = []

            if result.pose_landmarks:
                self.mp_drawing.draw_landmarks(output, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                h, w = output.shape[:2]
                for i in [0, 11, 12, 23, 24]:
                    lm = result.pose_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append([x, y])
                    cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
                for i in range(1, len(points)):
                    cv2.line(output, tuple(points[i-1]), tuple(points[i]), (255, 0, 0), 2)
            return output, np.array(points)

    def extract_spine_contour(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea, default=None)
        result = image.copy()
        if max_contour is not None:
            cv2.drawContours(result, [max_contour], -1, (0, 255, 0), 2)
        return result

    def calculate_cobb_angle(self, curve):
        if curve is None or len(curve) < 10:
            return None
        t, b = curve[:len(curve)//3], curve[2*len(curve)//3:]
        if len(t) > 1 and len(b) > 1:
            top_angle = math.atan(np.polyfit(t[:,1], t[:,0], 1)[0]) * 180 / math.pi
            bot_angle = math.atan(np.polyfit(b[:,1], b[:,0], 1)[0]) * 180 / math.pi
            return abs(top_angle - bot_angle)
        return None

    def enhance_xray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
