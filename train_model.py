"""
Skolyoz Tespit ve Analiz Sistemi - Model Eğitim Modülü
=====================================================
Bu script, skolyoz tespiti için model eğitir ve kaydeder.
Kullanım:
python train_model.py --train_dir veri_seti/train --val_dir veri_seti/val --model_type mobilenetv2 --epochs 50
"""
import os
import sys
import argparse
import tensorflow as tf
from datetime import datetime
from SkolyozModel import SkolyozModel

def parse_arguments():
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(description="Skolyoz tespiti için model eğitim")
    
    parser.add_argument("--train_dir", required=True, help="Eğitim veri seti dizini")
    parser.add_argument("--val_dir", required=True, help="Doğrulama veri seti dizini")
    parser.add_argument("--test_dir", help="Test veri seti dizini (opsiyonel)")
    
    parser.add_argument("--model_type", default="mobilenetv2", 
                        choices=["custom", "resnet50", "mobilenetv2", "efficientnet"],
                        help="Kullanılacak model mimarisi")
    
    parser.add_argument("--img_size", default=224, type=int, help="Görüntü boyutu")
    parser.add_argument("--epochs", default=50, type=int, help="Eğitim dönem sayısı")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch boyutu")
    parser.add_argument("--fine_tune", action="store_true", help="Transfer learning sonrası ince ayar yap")
    parser.add_argument("--fine_tune_epochs", default=20, type=int, help="İnce ayar dönem sayısı")
    
    parser.add_argument("--output_dir", default="models", help="Model kayıt dizini")
    
    return parser.parse_args()

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

def check_directories(args):
    """Veri seti dizinlerinin varlığını kontrol eder"""
    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"Eğitim dizini bulunamadı: {args.train_dir}")
    
    if not os.path.isdir(args.val_dir):
        raise FileNotFoundError(f"Doğrulama dizini bulunamadı: {args.val_dir}")
    
    if args.test_dir and not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"Test dizini bulunamadı: {args.test_dir}")
    
    # Çıktı dizinini oluştur
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Çıktı dizini oluşturuldu: {args.output_dir}")

def create_model_name(args):
    """Benzersiz model ismi oluşturur"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.model_type}_skolyoz_{args.img_size}px_{timestamp}.h5"

def main():
    """Ana program akışı"""
    # GPU yapılandırması
    configure_gpu()
    
    # Komut satırı argümanlarını işle
    args = parse_arguments()
    
    # Dizinleri kontrol et
    check_directories(args)
    
    print("=" * 50)
    print(f"Skolyoz Tespit ve Analiz Sistemi - Model Eğitimi Başlıyor")
    print(f"Model tipi: {args.model_type}")
    print(f"Görüntü boyutu: {args.img_size}x{args.img_size}")
    print(f"Eğitim dönemleri: {args.epochs}")
    print(f"Batch boyutu: {args.batch_size}")
    if args.fine_tune:
        print(f"İnce ayar yapılacak: Evet ({args.fine_tune_epochs} dönem)")
    print("=" * 50)
    
    # Model oluştur
    input_shape = (args.img_size, args.img_size, 3)
    model = SkolyozModel(model_type=args.model_type, input_shape=input_shape)
    
    # Veri jeneratörlerini oluştur
    train_generator, val_generator = model.create_data_generators(
        args.train_dir, args.val_dir, batch_size=args.batch_size
    )
    
    print(f"\nEğitim veri seti: {train_generator.samples} görüntü, {len(train_generator.class_indices)} sınıf")
    print(f"Doğrulama veri seti: {val_generator.samples} görüntü")
    print(f"Sınıf dağılımı: {train_generator.class_indices}")
    
    # Modeli oluştur
    print("\nModel oluşturuluyor...")
    model.build_model()
    model.model.summary()
    
    # Modeli eğit
    print(f"\nModel eğitimi başlıyor - {args.epochs} dönem...")
    history = model.train(train_generator, val_generator, epochs=args.epochs)
    
    # İnce ayar (fine-tuning)
    if args.fine_tune and args.model_type != "custom":
        print(f"\nİnce ayar yapılıyor - {args.fine_tune_epochs} dönem...")
        model.fine_tune(train_generator, val_generator, epochs=args.fine_tune_epochs)
    
    # Eğitim geçmişini görselleştir
    print("\nEğitim metriklerini görselleştirme...")
    model.plot_training_history()
    
    # Test veri seti varsa değerlendir
    if args.test_dir:
        print("\nTest veri seti üzerinde değerlendirme yapılıyor...")
        test_generator = model.create_test_generator(args.test_dir, batch_size=args.batch_size)
        evaluation_results = model.evaluate(test_generator)
        
        # Sonuçları yazdır
        print(f"\nDeğerlendirme sonuçları:")
        print(f"Test doğruluğu: {evaluation_results['test_accuracy']:.4f}")
        print(f"Test kaybı: {evaluation_results['test_loss']:.4f}")
    
    # Modeli kaydet
    model_filename = create_model_name(args)
    model_path = os.path.join(args.output_dir, model_filename)
    print(f"\nModel kaydediliyor: {model_path}")
    model.save_model_with_metadata(model_path)
    
    print("\nEğitim tamamlandı!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)