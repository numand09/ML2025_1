"""
SkolyozModel.py
Skolyoz Tespit ve Analiz Sistemi - Model
"""

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D
    from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
except ImportError:
    print("TensorFlow kütüphanesi bulunamadı veya import edilemedi.")
    print("Lütfen 'pip install tensorflow' komutu ile TensorFlow'u yükleyin.")

from SkolyozFramework import BaseModel

class SkolyozModel(BaseModel):
    def __init__(self, model_type="mobilenetv2", input_shape=(224, 224, 3), num_classes=3):
        super().__init__(input_shape, num_classes)
        self.model_type = model_type
        self.history = None
        self.class_names = ['Normal', 'C-Tipi Skolyoz', 'S-Tipi Skolyoz']
    
    def build_model(self):
        if self.model_type == "custom":
            self.model = self._create_custom_cnn_model()
        elif self.model_type in ["resnet50", "mobilenetv2", "efficientnet"]:
            self.model = self._create_transfer_learning_model()
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.model_type}")
        return self.model
    
    def _create_custom_cnn_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'), BatchNormalization(),
            MaxPooling2D((2, 2)), Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'), BatchNormalization(),
            MaxPooling2D((2, 2)), Dropout(0.25),
            
            Flatten(), Dense(512, activation='relu'), BatchNormalization(),
            Dropout(0.5), Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        return model
    
    def _create_transfer_learning_model(self):
        inputs = Input(shape=self.input_shape)
        
        if self.model_type == "resnet50":
            base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
        elif self.model_type == "mobilenetv2":
            base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
        elif self.model_type == "efficientnet":
            base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        return model
    
    def train(self, train_generator, val_generator, epochs=20):
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint(f"best_{self.model_type}_model.h5", save_best_only=True)
        ]
        
        self.history = self.model.fit(
            train_generator, validation_data=val_generator,
            epochs=epochs, callbacks=callbacks
        )
        return self.history
    
    def fine_tune(self, train_generator, val_generator, epochs=10):
        if self.model_type not in ["resnet50", "mobilenetv2", "efficientnet"]:
            print("İnce ayar sadece transfer öğrenme modelleri için geçerlidir")
            return None
        
        unfreeze_layers = {"resnet50": -30, "mobilenetv2": -20, "efficientnet": -15}
        for layer in self.model.layers[0].layers[unfreeze_layers.get(self.model_type, -20):]:
            layer.trainable = True
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(
            train_generator, validation_data=val_generator,
            epochs=epochs, callbacks=callbacks
        )
        return self.history
    
    def evaluate(self, test_generator):
        if self.model is None:
            raise ValueError("Değerlendirme için model gereklidir")
        
        test_loss, test_acc = self.model.evaluate(test_generator)
        print(f"Test Kaybı: {test_loss:.4f}")
        print(f"Test Doğruluğu: {test_acc:.4f}")
        
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        class_names = list(test_generator.class_indices.keys())
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Karmaşıklık matrisi
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Karmaşıklık Matrisi")
        plt.ylabel("Gerçek Etiket")
        plt.xlabel("Tahmin Edilen Etiket")
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # ROC eğrisi
        plt.figure(figsize=(8, 6))
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.title('ROC Eğrisi')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.show()
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        }
    
    def plot_training_history(self):
        if self.history is None:
            raise ValueError("Görselleştirme için eğitim geçmişi gereklidir")
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['sparse_categorical_accuracy'])
        plt.plot(self.history.history['val_sparse_categorical_accuracy'])
        plt.title('Model Doğruluğu')
        plt.ylabel('Doğruluk')
        plt.xlabel('Dönem')
        plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('Dönem')
        plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_model_with_metadata(self, model_path):
        self.save_model(model_path)
        with open(f"{os.path.splitext(model_path)[0]}_metadata.json", 'w') as f:
            json.dump({
                'class_names': self.class_names,
                'img_size': self.input_shape[:2],
                'model_type': self.model_type,
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, width_shift_range=0.2, 
            height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=self.input_shape[:2], batch_size=batch_size, 
            class_mode='sparse', shuffle=True
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir, target_size=self.input_shape[:2], batch_size=batch_size, 
            class_mode='sparse', shuffle=False
        )
        return train_generator, val_generator
    
    def create_test_generator(self, test_dir, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=self.input_shape[:2], batch_size=batch_size,
            class_mode='sparse', shuffle=False
        )
        return test_generator