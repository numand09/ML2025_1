"""
Skolyoz Tespit ve Sınıflandırma Entegre Sistemi
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, 
                           QStatusBar, QMessageBox, QGroupBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import mediapipe as mp
from skimage import io, transform, measure
import math


class VeriIsleme:
    """Temel veri işleme sınıfı"""
    
    @staticmethod
    def preprocess_image_for_model(image, target_size=(224, 224)):
        """Görüntüyü model için hazırlar"""
        if len(image.shape) == 2:  # Gri tonlamalı görüntü
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(image, target_size)
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)
    
    @staticmethod
    def detect_spine_in_xray(image):
        """X-ray görüntüsünde omurga tespiti yapar"""
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
                if len(x_values) > 0:
                    curve_points.append([np.mean(x_values), y])
            
            spine_curve = np.array(curve_points)
            for i in range(1, len(spine_curve)):
                cv2.line(result, 
                        tuple(spine_curve[i-1].astype(int)), 
                        tuple(spine_curve[i].astype(int)), 
                        (255, 0, 0), 2)
        
        return result, spine_curve
    
    @staticmethod
    def detect_spine_with_mediapipe(image):
        """MediaPipe kullanarak omurga tespiti yapar"""
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
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
    
    @staticmethod
    def calculate_cobb_angle(spine_curve):
        """Omurga eğrisi üzerinde Cobb açısını hesaplar"""
        if spine_curve is None or len(spine_curve) < 10:
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


class ModelOlusturucu(VeriIsleme):
    """Model oluşturma ve eğitim işlemleri için sınıf"""
    
    @staticmethod
    def create_custom_cnn_model(input_shape=(224, 224, 3), num_classes=3):
        """Özel CNN modeli oluşturur"""
        model = tf.keras.Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(), Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'), BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'), BatchNormalization(),
            MaxPooling2D((2, 2)), Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'), BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'), BatchNormalization(),
            MaxPooling2D((2, 2)), Dropout(0.25),
            
            Flatten(), Dense(512, activation='relu'), BatchNormalization(),
            Dropout(0.5), Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model
    
    @staticmethod
    def create_transfer_learning_model(base_model_name, input_shape=(224, 224, 3), num_classes=3):
        """Transfer öğrenme modeli oluşturur"""
        inputs = Input(shape=input_shape)
        
        if base_model_name == "resnet50":
            base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
        elif base_model_name == "mobilenetv2":
            base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
        else:
            raise ValueError(f"Desteklenmeyen model: {base_model_name}")
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32, img_size=(224, 224)):
        """Veri üreticilerini oluşturur"""
        train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=img_size, batch_size=batch_size, class_mode='sparse', shuffle=True
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir, target_size=img_size, batch_size=batch_size, class_mode='sparse', shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_model(self, model, train_generator, val_generator, epochs=30, fine_tune=False, base_model_name=None):
        """Modeli eğitir ve eğitim geçmişini döndürür"""
        if fine_tune and base_model_name:
            unfreeze_layers = {"resnet50": -30, "mobilenetv2": -20}
            for layer in model.layers[0].layers[unfreeze_layers.get(base_model_name, -20):]:
                layer.trainable = True
            
            model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint(f"best_model_{base_model_name if base_model_name else 'custom'}.h5", 
                          save_best_only=True)
        ]
        
        history = model.fit(train_generator, validation_data=val_generator, 
                          epochs=epochs, callbacks=callbacks)
        
        return model, history


class SkolyozSiniflandirici(VeriIsleme):
    """Skolyoz sınıflandırma işlemleri için sınıf"""
    
    def __init__(self, model_path=None):
        """Sınıflandırıcı için model yükler"""
        self.model = None
        self.class_names = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
        self.img_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"Model yüklendi: {model_path}")
            except Exception as e:
                print(f"Model yükleme hatası: {str(e)}")
                self._create_dummy_model()
        else:
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Test için basit bir model oluşturur"""
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print("Basit test modeli oluşturuldu")
    
    def predict(self, image):
        """Görüntü için sınıflandırma tahmini yapar"""
        if self.model is None:
            raise ValueError("Model yüklenmemiş")
        
        processed_img = self.preprocess_image_for_model(image, self.img_size)
        predictions = self.model.predict(processed_img)
        class_idx = np.argmax(predictions[0])
        class_name = self.class_names[class_idx]
        confidence = float(predictions[0][class_idx]) * 100
        
        return {
            'class_name': class_name,
            'confidence': confidence,
            'predictions': predictions[0]
        }
    
    def analyze_image(self, image_path):
        """Görüntüyü yükler, işler ve analiz sonuçlarını döndürür"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Omurga tespiti
            spine_image, spine_curve = self.detect_spine_in_xray(img)
            
            # MediaPipe ile tespit
            mediapipe_image, spine_points = self.detect_spine_with_mediapipe(img)
            
            # Cobb açısı hesaplama
            cobb_angle = self.calculate_cobb_angle(spine_curve) if spine_curve is not None and len(spine_curve) > 0 else 0.0
            
            # Sınıflandırma tahmini
            predictions = self.predict(img)
            
            results = {
                'class_name': predictions['class_name'],
                'confidence': predictions['confidence'],
                'cobb_angle': cobb_angle if cobb_angle else 0.0,
                'spine_image': spine_image,
                'mediapipe_image': mediapipe_image,
                'predictions': predictions['predictions']
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Görüntü analiz hatası: {str(e)}")


class AnalysisThread(QThread):
    """Analiz işlemlerini arka planda gerçekleştiren thread sınıfı"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, image_path, siniflandirici):
        super().__init__()
        self.image_path = image_path
        self.siniflandirici = siniflandirici
    
    def run(self):
        try:
            self.progress_signal.emit(10)
            
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self.progress_signal.emit(30)
            
            # Omurga tespiti yapılıyor
            spine_image, spine_curve = self.siniflandirici.detect_spine_in_xray(img)
            
            self.progress_signal.emit(50)
            
            # MediaPipe ile tespit yapılıyor
            mediapipe_image, spine_points = self.siniflandirici.detect_spine_with_mediapipe(img)
            
            self.progress_signal.emit(70)
            
            # Cobb açısı hesaplama
            cobb_angle = self.siniflandirici.calculate_cobb_angle(spine_curve) if spine_curve is not None else 0.0
            
            self.progress_signal.emit(80)
            
            # Sınıflandırma tahmini
            predictions = self.siniflandirici.predict(img)
            
            self.progress_signal.emit(100)
            
            results = {
                'class_name': predictions['class_name'],
                'confidence': predictions['confidence'],
                'cobb_angle': cobb_angle if cobb_angle else 0.0,
                'spine_image': spine_image,
                'mediapipe_image': mediapipe_image,
                'predictions': predictions['predictions'].tolist()
            }
            
            self.result_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib grafikleri için canvas sınıfı"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)


class SkolyozApp(QMainWindow):
    """Skolyoz tespit masaüstü uygulaması"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Skolyoz Tespit Uygulaması")
        self.setGeometry(100, 100, 1000, 700)
        
        self.current_image_path = None
        self.current_image = None
        self.siniflandirici = None
        self.model_paths = {
            "MobileNetV2": "best_model_mobilenetv2.h5",
            "ResNet50": "best_model_resnet50.h5",
            "Özel CNN": "best_model_custom.h5"
        }
        
        self.setup_ui()
        self.load_siniflandirici()
    
    def setup_ui(self):
        """Kullanıcı arayüzünü oluşturur"""
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Üst panel: Model seçimi, görüntü yükleme ve analiz
        top_panel = QHBoxLayout()
        
        # Model seçimi
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNetV2", "ResNet50", "Özel CNN"])
        self.model_combo.currentIndexChanged.connect(self.load_siniflandirici)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        
        # Görüntü yükleme
        controls_group = QGroupBox("Kontroller")
        controls_layout = QVBoxLayout()
        self.load_button = QPushButton("Röntgen Yükle")
        self.load_button.clicked.connect(self.load_image)
        self.analyze_button = QPushButton("Analiz Et")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.analyze_button)
        controls_layout.addWidget(self.progress_bar)
        controls_group.setLayout(controls_layout)
        
        top_panel.addWidget(model_group)
        top_panel.addWidget(controls_group)
        main_layout.addLayout(top_panel)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Orijinal görüntü sekmesi
        self.original_tab = QWidget()
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel("Henüz görüntü yüklenmedi")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_image_label)
        self.original_tab.setLayout(original_layout)
        
        # Sonuçlar sekmesi
        self.results_tab = QWidget()
        results_layout = QVBoxLayout()
        
        results_group = QGroupBox("Sonuçlar")
        results_inner_layout = QVBoxLayout()
        self.class_result_label = QLabel("Henüz analiz yapılmadı")
        self.confidence_label = QLabel("")
        self.cobb_angle_label = QLabel("")
        results_inner_layout.addWidget(self.class_result_label)
        results_inner_layout.addWidget(self.confidence_label)
        results_inner_layout.addWidget(self.cobb_angle_label)
        results_group.setLayout(results_inner_layout)
        
        self.prediction_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        
        results_layout.addWidget(results_group)
        results_layout.addWidget(self.prediction_canvas)
        self.results_tab.setLayout(results_layout)
        
        # Görüntü işleme sekmesi
        self.image_tab = QWidget()
        image_layout = QHBoxLayout()
        
        # Omurga tespiti
        spine_group = QGroupBox("Omurga Tespiti")
        spine_layout = QVBoxLayout()
        self.spine_image_label = QLabel("Analiz yapılmadı")
        self.spine_image_label.setAlignment(Qt.AlignCenter)
        spine_layout.addWidget(self.spine_image_label)
        spine_group.setLayout(spine_layout)
        
        # MediaPipe tespiti
        mediapipe_group = QGroupBox("MediaPipe Tespiti")
        mediapipe_layout = QVBoxLayout()
        self.mediapipe_image_label = QLabel("Analiz yapılmadı")
        self.mediapipe_image_label.setAlignment(Qt.AlignCenter)
        mediapipe_layout.addWidget(self.mediapipe_image_label)
        mediapipe_group.setLayout(mediapipe_layout)
        
        image_layout.addWidget(spine_group)
        image_layout.addWidget(mediapipe_group)
        self.image_tab.setLayout(image_layout)
        
        # Sekmeleri ekleme
        self.tab_widget.addTab(self.original_tab, "Orijinal Görüntü")
        self.tab_widget.addTab(self.image_tab, "Görüntü İşleme")
        self.tab_widget.addTab(self.results_tab, "Sonuçlar")
        
        main_layout.addWidget(self.tab_widget)
        
        # Durum çubuğu
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")
    
    def load_siniflandirici(self):
        """Seçilen modele göre sınıflandırıcıyı yükler"""
        try:
            model_name = self.model_combo.currentText()
            model_path = self.model_paths.get(model_name)
            
            if os.path.exists(model_path):
                self.siniflandirici = SkolyozSiniflandirici(model_path)
            else:
                self.siniflandirici = SkolyozSiniflandirici()  # Dummy model oluşturulacak
            
            self.status_bar.showMessage(f"Model yüklendi: {model_name}")
        
        except Exception as e:
            self.status_bar.showMessage(f"Model yükleme hatası: {str(e)}")
    
    def load_image(self):
        """Görüntü dosyası seçer ve yükler"""
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Röntgen Görüntüsü Seç", "", 
            "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        
        if image_path:
            try:
                self.current_image_path = image_path
                self.current_image = cv2.imread(image_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                self.display_image(self.current_image, self.original_image_label)
                self.analyze_button.setEnabled(True)
                self.status_bar.showMessage(f"Görüntü yüklendi: {os.path.basename(image_path)}")
                
            except Exception as e:
                self.status_bar.showMessage(f"Görüntü yükleme hatası: {str(e)}")
                QMessageBox.critical(self, "Hata", f"Görüntü yüklenemedi: {str(e)}")
    
    def display_image(self, img, label):
        """Görüntüyü etiket üzerinde gösterir"""
        h, w, c = img.shape
        bytes_per_line = c * w
        convert_to_qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_size = label.size()
        scaled_pixmap = QPixmap.fromImage(convert_to_qt_format).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
    
    def analyze_image(self):
        """Görüntü analizi başlatır"""
        if self.current_image_path is None or self.siniflandirici is None:
            self.status_bar.showMessage("Hata: Görüntü veya model yüklenmedi")
            return
        
        self.analyze_button.setEnabled(False)
        self.status_bar.showMessage("Analiz yapılıyor...")
        
        self.analysis_thread = AnalysisThread(
            self.current_image_path, 
            self.siniflandirici
        )
        
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.result_signal.connect(self.show_results)
        self.analysis_thread.error_signal.connect(self.show_error)
        
        self.analysis_thread.start()
    
    def update_progress(self, value):
        """İlerleme çubuğunu günceller"""
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        """Analiz sonuçlarını gösterir"""
        self.display_image(results['spine_image'], self.spine_image_label)
        self.display_image(results['mediapipe_image'], self.mediapipe_image_label)
        
        self.class_result_label.setText(f"Teşhis: {results['class_name']}")
        self.confidence_label.setText(f"Güven: %{results['confidence']:.2f}")
        self.cobb_angle_label.setText(f"Cobb Açısı: {results['cobb_angle']:.2f}°")
        
        # Tahmin grafiğini çizme
        self.prediction_canvas.axes.clear()
        x = np.arange(len(self.siniflandirici.class_names))
        bars = self.prediction_canvas.axes.bar(x, results['predictions'], color=['blue', 'orange', 'green'])
        self.prediction_canvas.axes.set_xticks(x)
        self.prediction_canvas.axes.set_xticklabels(self.siniflandirici.class_names)
        self.prediction_canvas.axes.set_ylim(0, 1)
        
        for bar, val in zip(bars, results['predictions']):
            height = bar.get_height()
            self.prediction_canvas.axes.text(
                bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}', ha='center', va='bottom'
            )
        
        self.prediction_canvas.draw()
        
        self.tab_widget.setCurrentIndex(2)  # Sonuçlar sekmesi
        self.status_bar.showMessage("Analiz tamamlandı")
        self.analyze_button.setEnabled(True)
    
    def show_error(self, error_message):
        """Hata mesajı gösterir"""
        self.status_bar.showMessage(f"Hata: {error_message}")
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında bir hata oluştu: {error_message}")
        
    def preprocess_image_for_model(image, target_size=(224, 224)):
        """Görüntüyü model için hazırlar"""
        if len(image.shape) == 2:  # Gri tonlamalı görüntü
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(image, target_size)
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def detect_spine_in_xray(image):
        """X-ray görüntüsünde omurga tespiti yapar"""
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
                if len(x_values) > 0:
                    curve_points.append([np.mean(x_values), y])
            
            spine_curve = np.array(curve_points)
            for i in range(1, len(spine_curve)):
                cv2.line(result, 
                        tuple(spine_curve[i-1].astype(int)), 
                        tuple(spine_curve[i].astype(int)), 
                        (255, 0, 0), 2)
        
        return result, spine_curve

    def detect_spine_with_mediapipe(image):
        """MediaPipe kullanarak omurga tespiti yapar"""
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
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

    def calculate_cobb_angle(spine_curve):
        """Omurga eğrisi üzerinde Cobb açısını hesaplar"""
        if spine_curve is None or len(spine_curve) < 10:
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

    def main():
        """Ana uygulama başlatma fonksiyonu"""
        app = QApplication(sys.argv)
        window = SkolyozApp()
        window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkolyozApp()  
    window.show()
    sys.exit(app.exec_())