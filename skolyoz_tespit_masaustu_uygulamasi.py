"""
Skolyoz Tespit Masaüstü Uygulaması
===============================
Bu uygulama, kullanıcıların röntgen görüntülerini yükleyip analiz etmelerini sağlar.
Skolyoz tespiti, sınıflandırması ve Cobb açısı tahmini yapar.
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, 
                           QStatusBar, QMessageBox, QGroupBox, QProgressBar, QSplitter)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Omurga tespiti ve Cobb açısı hesaplama için yardımcı fonksiyonlar
from skolyoz_veri_isleme import detect_spine_in_xray, calculate_cobb_angle, detect_spine_with_mediapipe


class AnalysisThread(QThread):
    """
    Analiz işlemlerini arka planda çalıştırmak için QThread sınıfı
    """
    progress_signal = pyqtSignal(int)  # İlerleme sinyali
    result_signal = pyqtSignal(dict)   # Sonuç sinyali
    error_signal = pyqtSignal(str)     # Hata sinyali
    
    def __init__(self, image_path, model, class_names, img_size):
        """
        Parametreler:
        image_path (str): Analiz edilecek görüntünün yolu
        model: Eğitilmiş Keras modeli
        class_names (list): Sınıf isimleri
        img_size (tuple): Model için görüntü boyutu
        """
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.class_names = class_names
        self.img_size = img_size
    
    def run(self):
        """
        Arka plan iş parçacığında çalışacak analiz işlemleri
        """
        try:
            # İlerleme: %10
            self.progress_signal.emit(10)
            
            # Görüntüyü yükleme
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # İlerleme: %20
            self.progress_signal.emit(20)
            
            # Omurga tespiti
            spine_image, spine_curve = detect_spine_in_xray(img)
            
            # İlerleme: %40
            self.progress_signal.emit(40)
            
            # Mediapipe ile omurga noktaları tespiti
            mediapipe_image, spine_points = detect_spine_with_mediapipe(img)
            
            # İlerleme: %60
            self.progress_signal.emit(60)
            
            # Cobb açısı hesaplama
            cobb_angle = None
            if spine_curve is not None and len(spine_curve) > 0:
                cobb_angle = calculate_cobb_angle(spine_curve)
            
            # İlerleme: %70
            self.progress_signal.emit(70)
            
            # Model için görüntüyü hazırlama
            model_img = cv2.resize(img, self.img_size)
            model_img = model_img / 255.0
            model_img = np.expand_dims(model_img, axis=0)
            
            # İlerleme: %80
            self.progress_signal.emit(80)
            
            # Sınıflandırma tahmini
            predictions = self.model.predict(model_img)
            class_idx = np.argmax(predictions[0])
            class_name = self.class_names[class_idx]
            confidence = float(predictions[0][class_idx]) * 100
            
            # İlerleme: %100
            self.progress_signal.emit(100)
            
            # Sonuçları oluşturma
            results = {
                'class_name': class_name,
                'confidence': confidence,
                'cobb_angle': cobb_angle,
                'spine_image': spine_image,
                'mediapipe_image': mediapipe_image,
                'predictions': predictions[0].tolist()
            }
            
            # Sonuçları gönderme
            self.result_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    """
    Matplotlib grafiklerini PyQt5 uygulamasına entegre etmek için kullanılan sınıf
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)


class SkolyozApp(QMainWindow):
    """
    Skolyoz Tespit Uygulaması ana sınıfı
    """
    def __init__(self):
        super().__init__()
        
        # Pencere başlığı ve boyutu
        self.setWindowTitle("Skolyoz Tespit Uygulaması")
        self.setGeometry(100, 100, 1200, 800)
        
        # Uygulama değişkenleri
        self.current_image_path = None
        self.current_image = None
        self.model = None
        self.class_names = []
        self.img_size = (224, 224)
        
        # Arayüz öğelerini ayarlama
        self.setup_ui()
        
        # Model yükleme işlemi
        self.load_model()
    
    def setup_ui(self):
        """
        Kullanıcı arayüzünü oluşturur
        """
        # Ana widget ve yerleşim
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Model ve kontrol seçenekleri için üst panel
        top_panel = QHBoxLayout()
        
        # Model seçimi
        model_group = QGroupBox("Model Seçimi")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNetV2 Modeli", "ResNet50 Modeli", "Özel CNN Modeli"])
        self.model_combo.currentIndexChanged.connect(self.load_model)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        
        # Görüntü yükleme düğmesi
        load_group = QGroupBox("Görüntü Yükleme")
        load_layout = QVBoxLayout()
        self.load_button = QPushButton("Röntgen Görüntüsü Yükle")
        self.load_button.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_button)
        load_group.setLayout(load_layout)
        
        # Analiz düğmesi
        analyze_group = QGroupBox("Analiz")
        analyze_layout = QVBoxLayout()
        self.analyze_button = QPushButton("Görüntüyü Analiz Et")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        analyze_layout.addWidget(self.analyze_button)
        analyze_group.setLayout(analyze_layout)
        
        # İlerleme çubuğu
        progress_group = QGroupBox("İlerleme")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)
        
        # Üst panele widget'ları ekleme
        top_panel.addWidget(model_group)
        top_panel.addWidget(load_group)
        top_panel.addWidget(analyze_group)
        top_panel.addWidget(progress_group)
        
        # Üst paneli ana yerleşime ekleme
        main_layout.addLayout(top_panel)
        
        # Görüntü ve sonuçlar için tab widget
        self.tab_widget = QTabWidget()
        
        # Orijinal görüntü sekmesi
        self.original_tab = QWidget()
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel("Henüz görüntü yüklenmedi")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_image_label)
        self.original_tab.setLayout(original_layout)
        
        # Omurga tespiti sekmesi
        self.spine_tab = QWidget()
        spine_layout = QVBoxLayout()
        self.spine_image_label = QLabel("Analiz yapılmadı")
        self.spine_image_label.setAlignment(Qt.AlignCenter)
        spine_layout.addWidget(self.spine_image_label)
        self.spine_tab.setLayout(spine_layout)
        
        # MediaPipe tespiti sekmesi
        self.mediapipe_tab = QWidget()
        mediapipe_layout = QVBoxLayout()
        self.mediapipe_image_label = QLabel("Analiz yapılmadı")
        self.mediapipe_image_label.setAlignment(Qt.AlignCenter)
        mediapipe_layout.addWidget(self.mediapipe_image_label)
        self.mediapipe_tab.setLayout(mediapipe_layout)
        
        # Sonuçlar sekmesi
        self.results_tab = QWidget()
        results_layout = QVBoxLayout()
        
        # Sınıflandırma sonuçları
        results_group = QGroupBox("Sınıflandırma Sonuçları")
        results_inner_layout = QVBoxLayout()
        self.class_result_label = QLabel("Henüz analiz yapılmadı")
        self.confidence_label = QLabel("")
        self.cobb_angle_label = QLabel("")
        results_inner_layout.addWidget(self.class_result_label)
        results_inner_layout.addWidget(self.confidence_label)
        results_inner_layout.addWidget(self.cobb_angle_label)
        results_group.setLayout(results_inner_layout)
        
        # Tahmin grafiği
        prediction_group = QGroupBox("Sınıflandırma Güveni")
        prediction_layout = QVBoxLayout()
        self.prediction_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        prediction_layout.addWidget(self.prediction_canvas)
        prediction_group.setLayout(prediction_layout)
        
        results_layout.addWidget(results_group)
        results_layout.addWidget(prediction_group)
        self.results_tab.setLayout(results_layout)
        
        # Sekmeleri ekleme
        self.tab_widget.addTab(self.original_tab, "Orijinal Görüntü")
        self.tab_widget.addTab(self.spine_tab, "Omurga Tespiti")
        self.tab_widget.addTab(self.mediapipe_tab, "MediaPipe Tespiti")
        self.tab_widget.addTab(self.results_tab, "Sonuçlar")
        
        # Tab widget'ı ana yerleşime ekleme
        main_layout.addWidget(self.tab_widget)
        
        # Durum çubuğu
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")
    
    def load_model(self):
        """
        Seçilen modeli yükler
        """
        try:
            model_index = self.model_combo.currentIndex() if hasattr(self, 'model_combo') else 0
            
            # Model dosya yolları (gerçek uygulamada bunlar gerçek model dosyaları olmalı)
            model_paths = {
                0: "models/mobilenetv2_scoliosis.h5",  # MobileNetV2
                1: "models/resnet50_scoliosis.h5",     # ResNet50
                2: "models/custom_cnn_scoliosis.h5"    # Özel CNN
            }
            
            # Sınıf isimleri
            self.class_names = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
            
            # Uygulama geliştirme aşamasında, model dosyası yoksa dummy model oluştur
            if not os.path.exists(model_paths[model_index]):
                # Uyarı göster
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(f"Uyarı: Model dosyası bulunamadı, test modu aktif.")
                
                # Basit model oluştur (sadece geliştirme için)
                inputs = tf.keras.Input(shape=(224, 224, 3))
                x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
                self.model = tf.keras.Model(inputs, outputs)
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            else:
                # Gerçek model yükleme
                self.model = load_model(model_paths[model_index])
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(f"Model başarıyla yüklendi: {self.model_combo.currentText()}")
        
        except Exception as e:
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Model yükleme hatası: {str(e)}")
            print(f"Model yükleme hatası: {str(e)}")
    
    def load_image(self):
        """
        Kullanıcının görüntü seçmesini sağlar
        """
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Röntgen Görüntüsü Seç", "", 
            "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        
        if image_path:
            try:
                # Görüntüyü yükleme
                self.current_image_path = image_path
                self.current_image = cv2.imread(image_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                # Görüntüyü gösterme
                self.display_image(self.current_image, self.original_image_label)
                
                # Analiz düğmesini etkinleştirme
                self.analyze_button.setEnabled(True)
                
                # Durum mesajı
                self.status_bar.showMessage(f"Görüntü yüklendi: {os.path.basename(image_path)}")
                
                # Diğer sekmeleri sıfırlama
                self.spine_image_label.setText("Analiz yapılmadı")
                self.mediapipe_image_label.setText("Analiz yapılmadı")
                self.class_result_label.setText("Henüz analiz yapılmadı")
                self.confidence_label.setText("")
                self.cobb_angle_label.setText("")
                self.prediction_canvas.axes.clear()
                self.prediction_canvas.draw()
                
            except Exception as e:
                self.status_bar.showMessage(f"Görüntü yükleme hatası: {str(e)}")
                QMessageBox.critical(self, "Hata", f"Görüntü yüklenemedi: {str(e)}")
    
    def display_image(self, img, label):
        """
        Görüntüyü belirtilen QLabel'da gösterir
        
        Parametreler:
        img (numpy.ndarray): Gösterilecek görüntü
        label (QLabel): Görüntünün gösterileceği etiket
        """
        h, w, c = img.shape
        bytes_per_line = c * w
        convert_to_qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Etiketin boyutuna göre yeniden boyutlandırma
        label_size = label.size()
        scaled_pixmap = QPixmap.fromImage(convert_to_qt_format).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
    
    def analyze_image(self):
        """
        Yüklenen görüntüyü analiz eder
        """
        if self.current_image_path is None or self.model is None:
            self.status_bar.showMessage("Hata: Görüntü veya model yüklenmedi")
            return
        
        # Analiz düğmesini devre dışı bırakma
        self.analyze_button.setEnabled(False)
        self.status_bar.showMessage("Analiz yapılıyor...")
        
        # Analiz iş parçacığını başlatma
        self.analysis_thread = AnalysisThread(
            self.current_image_path, 
            self.model, 
            self.class_names, 
            self.img_size
        )
        
        # Sinyalleri bağlama
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.result_signal.connect(self.show_results)
        self.analysis_thread.error_signal.connect(self.show_error)
        
        # İş parçacığını başlatma
        self.analysis_thread.start()
    
    def update_progress(self, value):
        """
        İlerleme çubuğunu günceller
        
        Parametreler:
        value (int): İlerleme değeri (0-100)
        """
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        """
        Analiz sonuçlarını gösterir
        
        Parametreler:
        results (dict): Analiz sonuçları
        """
        # Omurga tespiti görüntüsünü gösterme
        self.display_image(results['spine_image'], self.spine_image_label)
        
        # MediaPipe tespiti görüntüsünü gösterme
        self.display_image(results['mediapipe_image'], self.mediapipe_image_label)
        
        # Sonuç etiketlerini güncelleme
        self.class_result_label.setText(f"Teşhis: {results['class_name']}")
        self.confidence_label.setText(f"Güven: %{results['confidence']:.2f}")
        
        if results['cobb_angle'] is not None:
            self.cobb_angle_label.setText(f"Cobb Açısı: {results['cobb_angle']:.2f}°")
        else:
            self.cobb_angle_label.setText("Cobb Açısı: Hesaplanamadı")
        
        # Tahmin grafiğini çizme
        self.prediction_canvas.axes.clear()
        x = np.arange(len(self.class_names))
        bars = self.prediction_canvas.axes.bar(x, results['predictions'], color=['blue', 'orange', 'green'])
        self.prediction_canvas.axes.set_xticks(x)
        self.prediction_canvas.axes.set_xticklabels(self.class_names)
        self.prediction_canvas.axes.set_ylim(0, 1)
        self.prediction_canvas.axes.set_title("Sınıflandırma Sonuçları")
        
        # Çubukların üzerine değerleri yazma
        for bar, val in zip(bars, results['predictions']):
            height = bar.get_height()
            self.prediction_canvas.axes.text(
                bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}', ha='center', va='bottom'
            )
        
        self.prediction_canvas.draw()
        
        # Sonuçlar sekmesine geçme
        self.tab_widget.setCurrentIndex(3)  # Sonuçlar sekmesi
        
        # Durum mesajını güncelleme
        self.status_bar.showMessage("Analiz tamamlandı")
        
        # Analiz düğmesini yeniden etkinleştirme
        self.analyze_button.setEnabled(True)
    
    def show_error(self, error_message):
        """
        Hata mesajını gösterir
        
        Parametreler:
        error_message (str): Hata mesajı
        """
        self.status_bar.showMessage(f"Hata: {error_message}")
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında bir hata oluştu: {error_message}")
        
        # İlerleme çubuğunu sıfırlama
        self.progress_bar.setValue(0)
        
        # Analiz düğmesini yeniden etkinleştirme
        self.analyze_button.setEnabled(True)
    
    def export_results(self):
        """
        Analiz sonuçlarını dışa aktarır (PDF veya JSON)
        """
        # Bu işlevi uygulamanın ilerleyen sürümlerinde ekleyebilirsiniz
        pass
    
    def about_dialog(self):
        """
        Uygulama hakkında bilgi penceresi gösterir
        """
        QMessageBox.about(
            self, 
            "Hakkında", 
            "Skolyoz Tespit Uygulaması v1.0\n"
            "Bu uygulama, röntgen görüntülerinden skolyoz tespiti ve sınıflandırması yapar.\n"
            "Yapay zeka ile omurga eğriliğini analiz eder ve Cobb açısını tahmin eder."
        )

# Ana program
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkolyozApp()
    window.show()
    sys.exit(app.exec_())