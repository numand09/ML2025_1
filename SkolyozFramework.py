"""
Skolyoz Tespit ve Analiz Sistemi - Framework
===============================================
Temel sınıflar: Görüntü işleme, model, analiz ve kullanıcı arayüzü.
"""

import cv2, tensorflow as tf
from abc import ABC, abstractmethod

class BaseImageProcessor(ABC):
    def __init__(self, img_size=(224, 224)): self.img_size = img_size

    @abstractmethod
    def preprocess(self, image): pass

    def resize_image(self, image, size=None):
        return cv2.resize(image, size or self.img_size)

    def normalize_image(self, image): return image / 255.0

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None: raise ValueError(f"Yükleme hatası: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class BaseModel(ABC):
    def __init__(self, input_shape=(224,224,3), num_classes=3):
        self.input_shape, self.num_classes, self.model = input_shape, num_classes, None

    @abstractmethod
    def build_model(self): pass

    @abstractmethod
    def train(self, train_data, val_data, epochs=10): pass

    def save_model(self, path):
        if not self.model: raise ValueError("Model yok.")
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, image):
        if not self.model: raise ValueError("Model yüklenmemiş.")
        return self.model.predict(image)

class BaseAnalyzer(ABC):
    def __init__(self, model=None): self.model = model

    @abstractmethod
    def analyze(self, image): pass

    @abstractmethod
    def detect_spine(self, image): pass

    @abstractmethod
    def calculate_angle(self, spine_points): pass

    def set_model(self, model): self.model = model

class BaseUI(ABC):
    def __init__(self, analyzer=None): self.analyzer = analyzer

    @abstractmethod
    def setup_ui(self): pass

    @abstractmethod
    def display_results(self, results): pass

    def set_analyzer(self, analyzer): self.analyzer = analyzer
