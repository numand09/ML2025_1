"""
Skolyoz Tespit Masaüstü Uygulaması (Kısaltılmış Versiyon)
===============================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, 
                           QStatusBar, QMessageBox, QGroupBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, image_path, model, class_names, img_size):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.class_names = class_names
        self.img_size = img_size
    
    def run(self):
        try:
            self.progress_signal.emit(10)
            
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self.progress_signal.emit(40)
            
            # Basitleştirilmiş: Omurga tespiti ve MediaPipe işlevlerini temsil eden dummy fonksiyonlar
            spine_image = img.copy()  # Gerçek uygulamada detect_spine_in_xray(img) çağrılacak
            spine_curve = []  # Dummy veri
            
            mediapipe_image = img.copy()  # Gerçek uygulamada detect_spine_with_mediapipe(img) çağrılacak
            spine_points = []  # Dummy veri
            
            self.progress_signal.emit(60)
            
            # Basitleştirilmiş Cobb açısı hesaplama
            cobb_angle = 15.7  # Gerçek uygulamada calculate_cobb_angle(spine_curve) çağrılacak
            
            self.progress_signal.emit(80)
            
            # Model için görüntüyü hazırlama
            model_img = cv2.resize(img, self.img_size)
            model_img = model_img / 255.0
            model_img = np.expand_dims(model_img, axis=0)
            
            # Sınıflandırma tahmini
            predictions = self.model.predict(model_img)
            class_idx = np.argmax(predictions[0])
            class_name = self.class_names[class_idx]
            confidence = float(predictions[0][class_idx]) * 100
            
            self.progress_signal.emit(100)
            
            results = {
                'class_name': class_name,
                'confidence': confidence,
                'cobb_angle': cobb_angle,
                'spine_image': spine_image,
                'mediapipe_image': mediapipe_image,
                'predictions': predictions[0].tolist()
            }
            
            self.result_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)


class SkolyozApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Skolyoz Tespit Uygulaması")
        self.setGeometry(100, 100, 1000, 700)
        
        self.current_image_path = None
        self.current_image = None
        self.model = None
        self.class_names = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
        self.img_size = (224, 224)
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
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
        self.model_combo.currentIndexChanged.connect(self.load_model)
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
    
    def load_model(self):
        try:
            model_index = self.model_combo.currentIndex()
            
            # Basitleştirilmiş model yükleme (test için dummy model)
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            self.model = tf.keras.Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            
            self.status_bar.showMessage(f"Model yüklendi: {self.model_combo.currentText()}")
        
        except Exception as e:
            self.status_bar.showMessage(f"Model yükleme hatası: {str(e)}")
    
    def load_image(self):
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
        h, w, c = img.shape
        bytes_per_line = c * w
        convert_to_qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_size = label.size()
        scaled_pixmap = QPixmap.fromImage(convert_to_qt_format).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
    
    def analyze_image(self):
        if self.current_image_path is None or self.model is None:
            self.status_bar.showMessage("Hata: Görüntü veya model yüklenmedi")
            return
        
        self.analyze_button.setEnabled(False)
        self.status_bar.showMessage("Analiz yapılıyor...")
        
        self.analysis_thread = AnalysisThread(
            self.current_image_path, 
            self.model, 
            self.class_names, 
            self.img_size
        )
        
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.result_signal.connect(self.show_results)
        self.analysis_thread.error_signal.connect(self.show_error)
        
        self.analysis_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        self.display_image(results['spine_image'], self.spine_image_label)
        self.display_image(results['mediapipe_image'], self.mediapipe_image_label)
        
        self.class_result_label.setText(f"Teşhis: {results['class_name']}")
        self.confidence_label.setText(f"Güven: %{results['confidence']:.2f}")
        self.cobb_angle_label.setText(f"Cobb Açısı: {results['cobb_angle']:.2f}°")
        
        # Tahmin grafiğini çizme
        self.prediction_canvas.axes.clear()
        x = np.arange(len(self.class_names))
        bars = self.prediction_canvas.axes.bar(x, results['predictions'], color=['blue', 'orange', 'green'])
        self.prediction_canvas.axes.set_xticks(x)
        self.prediction_canvas.axes.set_xticklabels(self.class_names)
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
        self.status_bar.showMessage(f"Hata: {error_message}")
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında bir hata oluştu: {error_message}")
        self.progress_bar.setValue(0)
        self.analyze_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkolyozApp()
    window.show()
    sys.exit(app.exec_())