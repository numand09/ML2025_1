"""
Skolyoz Tespit ve Analiz Sistemi - Test Modülü
==============================================
Bu modül, sistemin bileşenlerini test etmek için örnek kod ve işlevler içerir.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from SkolyozFramework import BaseImageProcessor, BaseModel, BaseAnalyzer
from SkolyozImageProcessor import SkolyozImageProcessor
from SkolyozModel import SkolyozModel
from SkolyozAnalyzer import SkolyozAnalyzer

def test_image_processor(image_path):
    """Görüntü işleme modülünü test eder"""
    print("--- Görüntü İşlemci Testi ---")
    try:
        # Görüntü işlemci oluştur
        processor = SkolyozImageProcessor()
        
        # Görüntü yükle
        image = processor.load_image(image_path)
        print(f"Görüntü yüklendi: {image_path}, Boyut: {image.shape}")
        
        # Omurga tespiti
        print("Omurga tespiti yapılıyor...")
        spine_img, spine_curve = processor.detect_spine_in_xray(image)
        
        # MediaPipe ile tespit
        print("MediaPipe ile iskelet tespiti yapılıyor...")
        mp_img, mp_points = processor.detect_spine_with_mediapipe(image)
        
        # X-ray görüntüsünü iyileştir
        print("X-ray görüntüsü iyileştiriliyor...")
        enhanced = processor.enhance_xray(image)
        
        # Cobb açısı hesaplama
        if len(spine_curve) > 0:
            angle = processor.calculate_cobb_angle(spine_curve)
            print(f"Hesaplanan Cobb Açısı: {angle if angle else 'Belirlenemedi'}")
        
        # Sonuçları görselleştir
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Orijinal Görüntü")
        plt.imshow(image)
        
        plt.subplot(2, 2, 2)
        plt.title("İyileştirilmiş Görüntü")
        plt.imshow(enhanced)
        
        plt.subplot(2, 2, 3)
        plt.title("Omurga Tespiti")
        plt.imshow(spine_img)
        
        plt.subplot(2, 2, 4)
        plt.title("MediaPipe İskelet Tespiti")
        plt.imshow(mp_img)
        
        plt.tight_layout()
        plt.savefig("test_image_processor_results.png")
        plt.show()
        
        print("Görüntü işleme testi başarılı!")
        return spine_img, mp_img, enhanced
    
    except Exception as e:
        print(f"Görüntü işleme testi başarısız: {e}")
        return None, None, None

def test_model_creation():
    """Model oluşturma ve derleme işlemini test eder"""
    print("--- Model Oluşturma Testi ---")
    try:
        # Çeşitli modelleri oluştur ve mimariyi yazdır
        models = {
            "custom": SkolyozModel(model_type="custom"),
            "resnet50": SkolyozModel(model_type="resnet50"),
            "mobilenetv2": SkolyozModel(model_type="mobilenetv2"),
            "efficientnet": SkolyozModel(model_type="efficientnet")
        }
        
        for name, model in models.items():
            print(f"\n{name.upper()} modeli oluşturuluyor...")
            built_model = model.build_model()
            
            # Model özetini yazdır
            built_model.summary()
            
            print(f"{name} modeli başarıyla oluşturuldu ve derlendi.")
        
        print("\nModel oluşturma testi başarılı!")
        return models["mobilenetv2"]  # En hafif modeli döndür
    
    except Exception as e:
        print(f"Model oluşturma testi başarısız: {e}")
        return None

def prepare_dummy_data():
    """Test için örnek veri oluşturur"""
    print("--- Örnek Veri Hazırlanıyor ---")
    try:
        # Örnek görüntüler oluştur (3 sınıf, her biri için 5 görüntü)
        os.makedirs("dummy_data/train/normal", exist_ok=True)
        os.makedirs("dummy_data/train/c_type", exist_ok=True)
        os.makedirs("dummy_data/train/s_type", exist_ok=True)
        
        os.makedirs("dummy_data/val/normal", exist_ok=True)
        os.makedirs("dummy_data/val/c_type", exist_ok=True)
        os.makedirs("dummy_data/val/s_type", exist_ok=True)
        
        # Her sınıf için rastgele görüntüler oluştur
        for class_name in ["normal", "c_type", "s_type"]:
            for i in range(5):
                # Eğitim görüntüleri
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(f"dummy_data/train/{class_name}/img_{i}.jpg", img)
                
                # Doğrulama görüntüleri
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(f"dummy_data/val/{class_name}/img_{i}.jpg", img)
        
        print("Örnek veri hazırlandı.")
        return "dummy_data"
    
    except Exception as e:
        print(f"Örnek veri hazırlama hatası: {e}")
        return None

def test_model_training(model, epochs=2):
    """Model eğitimini test eder"""
    print("--- Model Eğitim Testi ---")
    try:
        # Örnek veri hazırla
        data_dir = prepare_dummy_data()
        if not data_dir:
            return None
        
        # Veri üreteçlerini oluştur
        train_generator, val_generator = model.create_data_generators(
            os.path.join(data_dir, "train"),
            os.path.join(data_dir, "val"),
            batch_size=2
        )
        
        print(f"Eğitim veri üreteci: {train_generator.n} görüntü, {train_generator.num_classes} sınıf")
        print(f"Doğrulama veri üreteci: {val_generator.n} görüntü")
        
        # Modeli eğit (çok kısa süre için)
        print(f"Model {epochs} dönem boyunca eğitiliyor...")
        history = model.train(train_generator, val_generator, epochs=epochs)
        
        # Eğitim geçmişini görselleştir
        model.plot_training_history()
        
        # Modeli kaydet
        model.save_model_with_metadata("test_model.h5")
        print("Model başarıyla kaydedildi: test_model.h5")
        
        print("Model eğitim testi başarılı!")
        return model
    
    except Exception as e:
        print(f"Model eğitim testi başarısız: {e}")
        return None

def test_analyzer(model, image_path):
    """Analizciyi test eder"""
    print("--- Analizci Testi ---")
    try:
        # Görüntü yükle
        processor = SkolyozImageProcessor()
        image = processor.load_image(image_path)
        
        # Analizci oluştur
        analyzer = SkolyozAnalyzer(model=model)
        
        # Görüntüyü analiz et
        print("Görüntü analiz ediliyor...")
        results = analyzer.analyze(image)
        
        # Sonuçları yazdır
        print("\nAnaliz Sonuçları:")
        print(f"Tahmin: {results['prediction']}")
        print(f"Güven Oranı: %{results['confidence']*100:.1f}")
        print(f"Cobb Açısı: {results['cobb_angle'] if results['cobb_angle'] else 'Belirlenemedi'}")
        
        # Şiddet belirle
        if results['cobb_angle']:
            severity = analyzer.get_skolyoz_severity(results['cobb_angle'])
            print(f"Skolyoz Şiddeti: {severity}")
        
        # Rapor oluştur
        report = analyzer.generate_report(results)
        print("\nÖneriler:")
        for recommendation in report['recommendations']:
            print(f"• {recommendation}")
        
        # Sonuçları görselleştir
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Orijinal Görüntü")
        plt.imshow(image)
        
        plt.subplot(2, 2, 2)
        plt.title("Omurga Tespiti")
        plt.imshow(results['spine_image'])
        
        plt.subplot(2, 2, 3)
        plt.title("MediaPipe İskelet Tespiti")
        plt.imshow(results['mediapipe_image'])
        
        plt.subplot(2, 2, 4)
        plt.title(f"Tahmin: {results['prediction']}")
        # Güven oranlarını göster
        bars = plt.bar(range(len(results['all_predictions'])), 
                        list(results['all_predictions'].values()))
        plt.xticks(range(len(results['all_predictions'])), 
                   list(results['all_predictions'].keys()), rotation=45)
        
        plt.tight_layout()
        plt.savefig("test_analyzer_results.png")
        plt.show()
        
        print("Analizci testi başarılı!")
        return analyzer
    
    except Exception as e:
        print(f"Analizci testi başarısız: {e}")
        return None

def main():
    """Test modülünün ana fonksiyonu"""
    print("=== SKOLYOZ TESPİT VE ANALİZ SİSTEMİ TEST MODÜLÜ ===\n")
    
    # Test edilecek görüntü yolu
    # Gerçek uygulamada burada bir skolyoz röntgen görüntüsü kullanılmalıdır
    image_path = input("Test edilecek görüntü dosyasının yolunu girin: ")
    
    if not os.path.exists(image_path):
        print(f"Hata: '{image_path}' bulunamadı.")
        return
    
    # Görüntü işlemciyi test et
    spine_img, mp_img, enhanced = test_image_processor(image_path)
    
    # Model oluşturmayı test et
    model = test_model_creation()
    if model:
        # Çok kısa bir eğitim testi (opsiyonel)
        choice = input("\nModel eğitimini test etmek istiyor musunuz? (e/h): ")
        if choice.lower() == 'e':
            model = test_model_training(model, epochs=1)
    
    # Analizciyi test et (eğer model varsa)
    if model:
        analyzer = test_analyzer(model, image_path)
    
    print("\n=== TEST TAMAMLANDI ===")

if __name__ == "__main__":
    main()