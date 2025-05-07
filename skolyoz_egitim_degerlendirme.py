"""
Skolyoz Modeli Eğitimi ve Değerlendirme - Özlü Versiyon
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import json

def create_data_generators(train_dir, val_dir, batch_size=32, img_size=(224, 224)):
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

def create_custom_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'), BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'), BatchNormalization(),
        MaxPooling2D((2, 2)), Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'), BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'), BatchNormalization(),
        MaxPooling2D((2, 2)), Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'), BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'), BatchNormalization(),
        MaxPooling2D((2, 2)), Dropout(0.25),
        
        Flatten(), Dense(512, activation='relu'), BatchNormalization(),
        Dropout(0.5), Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss=SparseCategoricalCrossentropy(),
                 metrics=[SparseCategoricalAccuracy()])
    return model

def create_transfer_learning_model(base_model_name, input_shape=(224, 224, 3), num_classes=3):
    inputs = Input(shape=input_shape)
    
    if base_model_name == "resnet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    elif base_model_name == "mobilenetv2":
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    elif base_model_name == "efficientnet":
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
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
                 loss=SparseCategoricalCrossentropy(),
                 metrics=[SparseCategoricalAccuracy()])
    return model

def fine_tune_model(model, base_model_name, train_generator, val_generator, epochs=10):
    # Temel modelin son katmanlarını açma
    unfreeze_layers = {"resnet50": -30, "mobilenetv2": -20, "efficientnet": -15}
    for layer in model.layers[0].layers[unfreeze_layers.get(base_model_name, -20):]:
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss=SparseCategoricalCrossentropy(),
                 metrics=[SparseCategoricalAccuracy()])
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    history = model.fit(train_generator, validation_data=val_generator, 
                       epochs=epochs, callbacks=callbacks)
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Dönem')
    plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Dönem')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Kaybı: {test_loss:.4f}")
    print(f"Test Doğruluğu: {test_acc:.4f}")
    
    predictions = model.predict(test_generator)
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

def save_model_with_metadata(model, model_path, class_names, img_size):
    model.save(model_path)
    
    with open(f"{os.path.splitext(model_path)[0]}_metadata.json", 'w') as f:
        json.dump({
            'class_names': class_names,
            'img_size': img_size,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)

def main():
    data_dir = "dataset"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    img_size = (224, 224)
    batch_size = 32
    num_classes = 3  # Normal, C-tipi, S-tipi
    
    print("Veri üretecileri oluşturuluyor...")
    train_generator, val_generator = create_data_generators(
        train_dir, val_dir, batch_size, img_size
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode='sparse', shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    print(f"Sınıflar: {class_names}")
    
    model_choice = "mobilenetv2"  # "custom", "resnet50", "mobilenetv2", "efficientnet"
    
    if model_choice == "custom":
        print("Özel CNN modeli oluşturuluyor...")
        model = create_custom_cnn_model(input_shape=(*img_size, 3), num_classes=num_classes)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
            ModelCheckpoint("best_custom_model.h5", save_best_only=True)
        ]
        
        history = model.fit(train_generator, validation_data=val_generator,
                          epochs=50, callbacks=callbacks)
    else:
        print(f"{model_choice} tabanlı transfer öğrenme modeli oluşturuluyor...")
        model = create_transfer_learning_model(
            model_choice, input_shape=(*img_size, 3), num_classes=num_classes
        )
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint(f"best_{model_choice}_initial.h5", save_best_only=True)
        ]
        
        history_initial = model.fit(train_generator, validation_data=val_generator,
                                  epochs=20, callbacks=callbacks)
        
        print("İnce ayar yapılıyor...")
        model, history = fine_tune_model(
            model, model_choice, train_generator, val_generator, epochs=30
        )
    
    plot_training_history(history)
    evaluate_model(model, test_generator)
    
    model_path = f"scoliosis_{model_choice}_model.h5"
    save_model_with_metadata(model, model_path, class_names, img_size)
    print(f"Model başarıyla kaydedildi: {model_path}")

if __name__ == "__main__":
    main()