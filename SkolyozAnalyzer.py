"""
Skolyoz Tespit ve Analiz Sistemi - Analizci
============================================
Bu modül, omurga görüntülerini analiz edip skolyoz tipini tespit eden ve
Cobb açısını hesaplayan analizci sınıfını içerir.
"""
import numpy as np
import cv2
import tensorflow as tf
from SkolyozFramework import BaseAnalyzer
from SkolyozImageProcessor import SkolyozImageProcessor

class SkolyozAnalyzer(BaseAnalyzer):
    """
    Skolyoz tespiti için analiz sınıfı.
    BaseAnalyzer'dan inheritance alır ve SkolyozImageProcessor'ı kullanır.
    """
    def __init__(self, model=None, class_names=None, img_size=(224, 224)):
        super().__init__(model)
        self.image_processor = SkolyozImageProcessor(img_size)
        self.class_names = class_names or ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
        self.img_size = img_size
    
    def analyze(self, image):
        """Görüntüyü analiz eder ve sonuçları döndürür"""
        if self.model is None:
            raise ValueError("Analiz için model yüklenmiş olmalıdır")
        
        # Omurga tespiti
        spine_image, spine_curve = self.detect_spine(image)
        
        # MediaPipe ile omurga noktalarını tespit et
        mediapipe_image, spine_points = self.image_processor.detect_spine_with_mediapipe(image)
        
        # Cobb açısı hesaplama
        cobb_angle = self.calculate_angle(spine_curve)
        
        # Model için görüntüyü hazırla
        preprocessed = self.image_processor.preprocess(image)
        
        # Skolyoz tipi tahmini
        predictions = self.model.predict(preprocessed)
        prediction_index = np.argmax(predictions[0])
        prediction_class = self.class_names[prediction_index]
        confidence = float(predictions[0][prediction_index])
        
        return {
            'spine_image': spine_image,
            'mediapipe_image': mediapipe_image,
            'spine_curve': spine_curve,
            'spine_points': spine_points,
            'cobb_angle': cobb_angle,
            'prediction': prediction_class,
            'confidence': confidence,
            'all_predictions': {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        }
    
    def detect_spine(self, image):
        """Omurga eğrisini tespit eder"""
        # Önce X-ray işleme yöntemini dene
        xray_img, xray_curve = self.image_processor.detect_spine_in_xray(image)
        
        # Eğer yeterli nokta bulunamazsa MediaPipe ile dene
        if len(xray_curve) < 5:
            mp_img, mp_curve = self.image_processor.detect_spine_with_mediapipe(image)
            return mp_img, mp_curve
        
        return xray_img, xray_curve
    
    def calculate_angle(self, spine_points):
        """Omurga eğrisi üzerinden Cobb açısını hesaplar"""
        if spine_points is None or len(spine_points) < 10:
            return None
        
        # Cobb açısı hesaplama
        try:
            return self.image_processor.calculate_cobb_angle(spine_points)
        except Exception as e:
            print(f"Cobb açısı hesaplanırken hata: {e}")
            return None
    
    def get_skolyoz_severity(self, cobb_angle):
        """Cobb açısına göre skolyoz şiddetini belirler"""
        if cobb_angle is None:
            return "Belirlenemedi"
        elif cobb_angle < 10:
            return "Normal (Skolyoz Yok)"
        elif cobb_angle < 25:
            return "Hafif Skolyoz"
        elif cobb_angle < 45:
            return "Orta Şiddetli Skolyoz"
        else:
            return "Şiddetli Skolyoz"
    
    def enhance_image(self, image):
        """Analiz için görüntüyü iyileştirir"""
        return self.image_processor.enhance_xray(image)
    
    def detect_and_highlight_vertebrae(self, image):
        """Omurları tespit edip işaretler"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Kontrast artırma
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptif eşik değeri
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Kenar tespiti
        edges = cv2.Canny(thresh, 30, 200)
        
        # Konturlar
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Vertebra olabilecek konturları filtrele
        vertebrae = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:  # Boyut filtresi
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:  # Şekil filtresi
                    vertebrae.append(cnt)
        
        # Sonuç görüntüsü
        result = image.copy()
        cv2.drawContours(result, vertebrae, -1, (0, 255, 255), 2)
        
        return result, vertebrae
    
    def generate_report(self, results):
        """Analiz sonuçlarından detaylı rapor oluşturur"""
        cobb_angle = results.get('cobb_angle')
        severity = self.get_skolyoz_severity(cobb_angle)
        prediction = results.get('prediction')
        confidence = results.get('confidence', 0) * 100
        
        report = {
            'prediction': prediction,
            'confidence': f"{confidence:.1f}%",
            'cobb_angle': f"{cobb_angle:.1f}°" if cobb_angle else "Belirlenemedi",
            'severity': severity,
            'recommendations': self._get_recommendations(severity, prediction)
        }
        
        return report
    
    def _get_recommendations(self, severity, skolyoz_type):
        """Skolyoz tipi ve şiddetine göre öneriler oluşturur"""
        recommendations = []
        
        if severity == "Normal (Skolyoz Yok)":
            recommendations = [
                "Düzenli fiziksel aktivite",
                "Duruş kontrolü",
                "Periyodik kontroller (yılda bir)"
            ]
        elif severity == "Hafif Skolyoz":
            recommendations = [
                "Fizik tedavi ve rehabilitasyon",
                "Duruş egzersizleri",
                "6 ayda bir kontrol",
                "Korse tedavisi değerlendirmesi"
            ]
        elif severity == "Orta Şiddetli Skolyoz":
            recommendations = [
                "Uzman ortopedist/omurga cerrahı değerlendirmesi",
                "Korse tedavisi",
                "Özel fizik tedavi programı",
                "3 ayda bir kontrol"
            ]
        elif severity == "Şiddetli Skolyoz":
            recommendations = [
                "Acil uzman değerlendirmesi",
                "Cerrahi tedavi değerlendirmesi",
                "Düzenli takip",
                "Ağrı yönetimi"
            ]
        
        # Skolyoz tipine göre ek öneriler
        if skolyoz_type == "C-Tipi Skolyoz":
            recommendations.append("C-tipi eğriliğe yönelik özel egzersizler")
        elif skolyoz_type == "S-Tipi Skolyoz":
            recommendations.append("S-tipi eğriliğe yönelik özel egzersizler")
            recommendations.append("Daha sık kontrol (S-tipi eğriler daha hızlı ilerleyebilir)")
        
        return recommendations