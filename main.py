"""
Skolyoz Tespit ve Analiz Sistemi - Ana Uygulama (PyQt5 tabanlı)
===============================================
Bu dosya, tüm modülleri bir araya getirip uygulamayı başlatır.
"""
import os
import sys
import argparse
import tensorflow as tf
from SkolyozAnalyzer import SkolyozAnalyzer
from SkolyozModel import SkolyozModel
from SkolyozUI import SkolyozUI  # PyQt5 tabanlı UI sınıfı

# GPU bellek ayarları
def configure_gpu():
    """GPU kullanımı için bellek ayarlarını yapılandırır"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU bulundu ve yapılandırıldı.")
    except Exception as e:
        print(f"GPU yapılandırma hatası: {e}")

def create_analyzer(model_path=None):
    """Analizci oluşturur"""
    analyzer = None
    try:
        if model_path and os.path.exists(model_path):
            # Model yükle
            print(f"Model yükleniyor: {model_path}")
            model = SkolyozModel()
            model.load_model(model_path)
            # Analizci oluştur
            analyzer = SkolyozAnalyzer(model=model)
            print("Model başarıyla yüklendi ve analizci oluşturuldu.")
        else:
            # Model olmadan analizci oluştur
            analyzer = SkolyozAnalyzer()
            print("Analizci oluşturuldu (model yüklenmedi).")
    except Exception as e:
        print(f"Analizci oluşturma hatası: {e}")
        # Hata olsa bile boş bir analizci döndür
        analyzer = SkolyozAnalyzer()
    
    return analyzer

def main():
    """Ana uygulama fonksiyonu"""
    # Komut satırı argümanlarını işle
    parser = argparse.ArgumentParser(description="Skolyoz Tespit ve Analiz Sistemi")
    parser.add_argument("--model", "-m", help="Eğitilmiş model dosyası yolu")
    parser.add_argument("--image", "-i", help="Analiz edilecek görüntü dosyası")
    args = parser.parse_args()
    
    # GPU yapılandırması
    configure_gpu()
    
    # Analizci oluştur
    analyzer = create_analyzer(args.model)
    
    # SkolyozUI sınıfını kullanarak arayüzü başlat
    # Not: SkolyozUI kendi içinde QApplication oluşturup yönetiyor
    ui = SkolyozUI(analyzer)
    
    # Eğer görüntü belirtilmişse otomatik olarak yükle
    if args.image and os.path.exists(args.image):
        # ui.load_image metodu direkt olarak dosya yolunu
        # kabul etmiyor, o yüzden yardımcı bir yöntem eklenebilir
        try:
            ui.image_path_edit.setText(args.image)
            ui.load_image()
        except Exception as e:
            print(f"Görüntü yükleme hatası: {e}")
    
    # Uygulamayı başlat
    ui.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Uygulama başlatılırken hata oluştu: {e}")
        # Hata ayrıntılarını yazdır
        import traceback
        traceback.print_exc()