import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QAction, QFileDialog, 
                            QMessageBox, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, 
                            QTabWidget, QFrame, QListWidget, QLineEdit, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import io
import matplotlib.pyplot as plt

from SkolyozFramework import BaseUI
from SkolyozAnalyzer import SkolyozAnalyzer
from SkolyozModel import SkolyozModel

class SkolyozUI(BaseUI):
    def __init__(self, analyzer=None):
        super().__init__(analyzer)
        self.current_image = None
        self.current_results = None
        self.results_images = {}
        self.setup_ui()
    
    def setup_ui(self):
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("Skolyoz Tespit ve Analiz Sistemi")
        self.window.setGeometry(100, 100, 1200, 800)
        
        # Ana menü
        menubar = self.window.menuBar()
        file_menu = menubar.addMenu("Dosya")
        model_menu = menubar.addMenu("Model")
        help_menu = menubar.addMenu("Yardım")
        
        # Dosya menüsü
        load_img_action = QAction("Görüntü Yükle", self.window)
        load_img_action.triggered.connect(self.load_image)
        save_report_action = QAction("Rapor Kaydet", self.window)
        save_report_action.triggered.connect(self.save_report)
        exit_action = QAction("Çıkış", self.window)
        exit_action.triggered.connect(self.window.close)
        
        file_menu.addAction(load_img_action)
        file_menu.addAction(save_report_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # Model menüsü
        load_model_action = QAction("Model Yükle", self.window)
        load_model_action.triggered.connect(self.load_model)
        model_menu.addAction(load_model_action)
        
        # Yardım menüsü
        about_action = QAction("Hakkında", self.window)
        about_action.triggered.connect(self.show_about)
        help_action = QAction("Kullanım Kılavuzu", self.window)
        help_action.triggered.connect(self.show_help)
        
        help_menu.addAction(about_action)
        help_menu.addAction(help_action)
        
        # Ana widget ve layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Sol panel (Görüntü ve kontroller)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Dosya işlemleri
        file_group = QGroupBox("Dosya İşlemleri")
        file_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        browse_button = QPushButton("Görsel Seç")
        browse_button.clicked.connect(self.load_image)
        
        file_layout.addWidget(self.image_path_edit)
        file_layout.addWidget(browse_button)
        file_group.setLayout(file_layout)
        
        # Görüntü alanı
        image_group = QGroupBox("Görüntü")
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        
        # Kontrol düğmeleri
        control_layout = QHBoxLayout()
        analyze_button = QPushButton("Analiz Et")
        analyze_button.clicked.connect(self.analyze_image)
        enhance_button = QPushButton("Görüntüyü İyileştir")
        enhance_button.clicked.connect(self.enhance_image)
        clear_button = QPushButton("Temizle")
        clear_button.clicked.connect(self.clear_results)
        
        control_layout.addWidget(analyze_button)
        control_layout.addWidget(enhance_button)
        control_layout.addWidget(clear_button)
        
        # Sol paneli birleştir
        left_layout.addWidget(file_group)
        left_layout.addWidget(image_group, 1)
        left_layout.addLayout(control_layout)
        left_panel.setLayout(left_layout)
        
        # Sağ panel (Sonuçlar)
        self.tab_widget = QTabWidget()
        
        # Sonuç tab
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        
        self.result_title = QLabel("Henüz analiz yapılmadı")
        self.result_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        results_layout.addWidget(self.result_title)
        
        # Sonuç detayları
        details_group = QGroupBox("Detaylar")
        details_layout = QVBoxLayout()
        
        # Skolyoz tipi
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Skolyoz Tipi:"))
        self.type_label = QLabel("-")
        type_layout.addWidget(self.type_label)
        details_layout.addLayout(type_layout)
        
        # Güven oranı
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Güven Oranı:"))
        self.confidence_label = QLabel("-")
        conf_layout.addWidget(self.confidence_label)
        details_layout.addLayout(conf_layout)
        
        # Cobb açısı
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Cobb Açısı:"))
        self.angle_label = QLabel("-")
        angle_layout.addWidget(self.angle_label)
        details_layout.addLayout(angle_layout)
        
        # Şiddet
        severity_layout = QHBoxLayout()
        severity_layout.addWidget(QLabel("Şiddet:"))
        self.severity_label = QLabel("-")
        severity_layout.addWidget(self.severity_label)
        details_layout.addLayout(severity_layout)
        
        details_group.setLayout(details_layout)
        results_layout.addWidget(details_group)
        
        # Öneriler
        recommendations_group = QGroupBox("Öneriler")
        recommendations_layout = QVBoxLayout()
        self.recommendations_list = QListWidget()
        recommendations_layout.addWidget(self.recommendations_list)
        recommendations_group.setLayout(recommendations_layout)
        results_layout.addWidget(recommendations_group)
        
        results_tab.setLayout(results_layout)
        
        # Görüntü işleme tabı
        image_proc_tab = QWidget()
        self.proc_layout = QHBoxLayout()
        image_proc_tab.setLayout(self.proc_layout)
        
        # Tabları ekle
        self.tab_widget.addTab(results_tab, "Analiz Sonuçları")
        self.tab_widget.addTab(image_proc_tab, "Görüntü İşleme")
        
        # Ana layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.tab_widget, 1)
        
        central_widget.setLayout(main_layout)
        self.window.setCentralWidget(central_widget)
        
        # Durum çubuğu
        self.status_bar = self.window.statusBar()
        self.status_bar.showMessage("Hazır")
        
        # Model kontrolü
        if self.analyzer is None or self.analyzer.model is None:
            self.status_bar.showMessage("Uyarı: Model yüklü değil! 'Model > Model Yükle' menüsünü kullanın.")
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, 
            "Görüntü Seç", 
            "", 
            "Görüntü Dosyaları (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            try:
                self.image_path_edit.setText(file_path)
                self.current_image = cv2.imread(file_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                self.display_image(self.current_image)
                self.status_bar.showMessage(f"Görüntü yüklendi: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self.window, "Hata", f"Görüntü yüklenirken hata oluştu: {str(e)}")
                self.status_bar.showMessage("Hata: Görüntü yüklenemedi")
    
    def display_image(self, image, label=None):
        if image is None:
            return
        
        if label is None:
            label = self.image_label
        
        h, w = image.shape[:2]
        max_size = 500
        
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        h, w = image.shape[:2]
        q_img = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)
    
    def analyze_image(self):
        if self.current_image is None:
            QMessageBox.warning(self.window, "Uyarı", "Lütfen önce bir görüntü yükleyin.")
            return
        
        if self.analyzer is None or self.analyzer.model is None:
            QMessageBox.warning(self.window, "Uyarı", "Lütfen önce bir model yükleyin.")
            return
        
        try:
            self.status_bar.showMessage("Analiz yapılıyor...")
            QApplication.processEvents()
            
            results = self.analyzer.analyze(self.current_image)
            self.current_results = results
            
            self.display_results(results)
            
            self.status_bar.showMessage("Analiz tamamlandı.")
        except Exception as e:
            QMessageBox.critical(self.window, "Hata", f"Analiz sırasında hata oluştu: {str(e)}")
            self.status_bar.showMessage("Hata: Analiz yapılamadı")
    
    def display_results(self, results):
        if results is None:
            return
        
        prediction = results.get('prediction', 'Belirsiz')
        confidence = results.get('confidence', 0) * 100
        self.type_label.setText(prediction)
        self.confidence_label.setText(f"%{confidence:.1f}")
        
        cobb_angle = results.get('cobb_angle')
        if cobb_angle is not None:
            severity = self.analyzer.get_skolyoz_severity(cobb_angle)
            self.angle_label.setText(f"{cobb_angle:.1f}°")
            self.severity_label.setText(severity)
            self.result_title.setText(f"{prediction} - {severity}")
        else:
            self.angle_label.setText("Belirlenemedi")
            self.severity_label.setText("Belirlenemedi")
            self.result_title.setText(prediction)
        
        self.recommendations_list.clear()
        report = self.analyzer.generate_report(results)
        for recommendation in report.get('recommendations', []):
            self.recommendations_list.addItem(f"• {recommendation}")
        
        self.results_images = {
            'spine_image': results.get('spine_image'),
            'mediapipe_image': results.get('mediapipe_image')
        }
        
        self._update_image_processing_tab()
    
    def _update_image_processing_tab(self):
        # Temizlik
        for i in reversed(range(self.proc_layout.count())): 
            self.proc_layout.itemAt(i).widget().setParent(None)
        
        if not self.results_images:
            self.proc_layout.addWidget(QLabel("Henüz görüntü işleme sonuçları yok"))
            return
        
        if 'spine_image' in self.results_images and self.results_images['spine_image'] is not None:
            spine_group = QGroupBox("Omurga Tespiti")
            spine_layout = QVBoxLayout()
            spine_label = QLabel()
            spine_layout.addWidget(spine_label)
            spine_group.setLayout(spine_layout)
            self.proc_layout.addWidget(spine_group)
            self.display_image(self.results_images['spine_image'], spine_label)
        
        if 'mediapipe_image' in self.results_images and self.results_images['mediapipe_image'] is not None:
            mp_group = QGroupBox("MediaPipe İskelet Tespiti")
            mp_layout = QVBoxLayout()
            mp_label = QLabel()
            mp_layout.addWidget(mp_label)
            mp_group.setLayout(mp_layout)
            self.proc_layout.addWidget(mp_group)
            self.display_image(self.results_images['mediapipe_image'], mp_label)
    
    def enhance_image(self):
        if self.current_image is None:
            QMessageBox.warning(self.window, "Uyarı", "Lütfen önce bir görüntü yükleyin.")
            return
        
        if self.analyzer is None:
            QMessageBox.warning(self.window, "Uyarı", "Analizci yüklü değil.")
            return
        
        try:
            enhanced = self.analyzer.enhance_image(self.current_image)
            self.current_image = enhanced
            self.display_image(enhanced)
            self.status_bar.showMessage("Görüntü iyileştirildi.")
        except Exception as e:
            QMessageBox.critical(self.window, "Hata", f"Görüntü iyileştirme hatası: {str(e)}")
    
    def clear_results(self):
        self.current_results = None
        self.results_images = {}
        
        self.result_title.setText("Henüz analiz yapılmadı")
        self.type_label.setText("-")
        self.confidence_label.setText("-")
        self.angle_label.setText("-")
        self.severity_label.setText("-")
        
        self.recommendations_list.clear()
        self._update_image_processing_tab()
        
        self.status_bar.showMessage("Sonuçlar temizlendi.")
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Model Seç",
            "",
            "TensorFlow Model (*.h5)"
        )
        
        if file_path:
            try:
                self.status_bar.showMessage("Model yükleniyor...")
                QApplication.processEvents()
                
                model = SkolyozModel()
                model.load_model(file_path)
                
                if self.analyzer is None:
                    self.analyzer = SkolyozAnalyzer(model=model)
                else:
                    self.analyzer.set_model(model)
                
                self.status_bar.showMessage(f"Model yüklendi: {os.path.basename(file_path)}")
                QMessageBox.information(self.window, "Başarılı", "Model başarıyla yüklendi.")
                
            except Exception as e:
                QMessageBox.critical(self.window, "Hata", f"Model yüklenirken hata oluştu: {str(e)}")
                self.status_bar.showMessage("Hata: Model yüklenemedi")
    
    def save_report(self):
        if self.current_results is None:
            QMessageBox.warning(self.window, "Uyarı", "Önce analiz yapmalısınız.")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.window,
            "Raporu Kaydet",
            "",
            "PDF Dosyası (*.pdf);;Text Dosyası (*.txt)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith(".pdf"):
                self._save_pdf_report(file_path)
            else:
                self._save_text_report(file_path)
                
            self.status_bar.showMessage(f"Rapor kaydedildi: {os.path.basename(file_path)}")
            QMessageBox.information(self.window, "Başarılı", "Rapor başarıyla kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self.window, "Hata", f"Rapor kaydedilirken hata oluştu: {str(e)}")
    
    def _save_pdf_report(self, file_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("Skolyoz Analiz Raporu", styles['Heading1']))
        elements.append(Spacer(1, 0.5*inch))
        
        import datetime
        elements.append(Paragraph(f"Tarih: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph("Hasta Bilgileri:", styles['Heading2']))
        data = [["Hasta Adı:", "..."], ["Hasta No:", "..."], ["Doğum Tarihi:", "..."]]
        t = Table(data, colWidths=[100, 400])
        t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph("Analiz Sonuçları:", styles['Heading2']))
        
        report = self.analyzer.generate_report(self.current_results)
        
        data = [
            ["Skolyoz Tipi:", report['prediction']],
            ["Güven Oranı:", report['confidence']],
            ["Cobb Açısı:", report['cobb_angle']],
            ["Şiddet:", report['severity']]
        ]
        t = Table(data, colWidths=[100, 400])
        t.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph("Öneriler:", styles['Heading2']))
        for recommendation in report['recommendations']:
            elements.append(Paragraph(f"• {recommendation}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        if self.current_image is not None:
            img_bytes = io.BytesIO()
            pil_img = Image.fromarray(self.current_image)
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            elements.append(Paragraph("Orijinal Görüntü:", styles['Heading3']))
            img = Image(img_bytes, width=4*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        if 'spine_image' in self.results_images and self.results_images['spine_image'] is not None:
            img_bytes = io.BytesIO()
            pil_img = Image.fromarray(self.results_images['spine_image'])
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            elements.append(Paragraph("Omurga Tespiti:", styles['Heading3']))
            img = Image(img_bytes, width=4*inch, height=4*inch)
            elements.append(img)
        
        doc.build(elements)
    
    def _save_text_report(self, file_path):
        import datetime
        
        report = self.analyzer.generate_report(self.current_results)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== SKOLYOZ ANALİZ RAPORU ===\n\n")
            f.write(f"Tarih: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            
            f.write("-- Hasta Bilgileri --\n")
            f.write("Hasta Adı: ...\n")
            f.write("Hasta No: ...\n")
            f.write("Doğum Tarihi: ...\n\n")
            
            f.write("-- Analiz Sonuçları --\n")
            f.write(f"Skolyoz Tipi: {report['prediction']}\n")
            f.write(f"Güven Oranı: {report['confidence']}\n")
            f.write(f"Cobb Açısı: {report['cobb_angle']}\n")
            f.write(f"Şiddet: {report['severity']}\n\n")
            
            f.write("-- Öneriler --\n")
            for recommendation in report['recommendations']:
                f.write(f"• {recommendation}\n")
    
    def show_about(self):
        QMessageBox.information(
            self.window,
            "Hakkında", 
            "Skolyoz Tespit ve Analiz Sistemi\n"
            "Sürüm 1.0\n\n"
            "Bu uygulama, röntgen görüntülerinden skolyoz tespiti ve analizi için geliştirilmiştir.\n"
            "Derin öğrenme teknikleri kullanılarak skolyoz tipi belirlenir ve Cobb açısı hesaplanır."
        )
    
    def show_help(self):
        help_text = """
Kullanım Kılavuzu:

1. 'Dosya > Görüntü Yükle' menüsü ile analiz edilecek görüntüyü seçin.
2. 'Model > Model Yükle' menüsü ile eğitilmiş modeli yükleyin.
3. 'Analiz Et' düğmesine tıklayarak görüntü analizini başlatın.
4. Sonuçlar sağ panelde gösterilecektir.
5. 'Dosya > Rapor Kaydet' menüsü ile sonuçları PDF olarak kaydedebilirsiniz.

Notlar:
- En iyi sonuçlar için PA (posterior-anterior) röntgen görüntüleri önerilir.
- Görüntüleri JPEG, PNG veya BMP formatında yükleyebilirsiniz.
- 'Görüntüyü İyileştir' düğmesi kontrastı artırarak omurga tespitini iyileştirebilir.
"""
        msg = QMessageBox()
        msg.setWindowTitle("Kullanım Kılavuzu")
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())