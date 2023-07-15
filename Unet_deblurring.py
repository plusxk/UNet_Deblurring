import os
import tensorflow as tf
from tensorflow.keras import layers, metrics, losses
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import Adam
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import random
import PIL
import cv2


with tf.device('/device:GPU:0'):
    batch_size = 2
    epochs = 10
    learning_rate = 1e-4

    # 設置資料夾路徑和子資料夾名稱
    # folder_path = 'D:\\gopro\\train\\blur'
    # blur_folder = 'blur'
    # sharp_folder = 'sharp'


    # 原圖目錄位置
    input_dir = "D:/gopro/train/blur/"
    # 目標圖遮罩(Mask)的目錄位置
    target_dir = "D:/gopro/train/sharp/"
    # 超參數
    img_size = (360, 640)
    num_classes = 1

    # 取得原圖檔案路徑
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )

    # 取得目標圖遮罩的檔案路徑
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    print("樣本數:", len(input_img_paths))
    class OxfordPets(tf.keras.utils.Sequence):
        """Helper to iterate over the data (as Numpy arrays)."""

        def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
            self.batch_size = batch_size
            self.img_size = img_size
            self.input_img_paths = input_img_paths
            self.target_img_paths = target_img_paths

        def __len__(self):
            return len(self.target_img_paths) // self.batch_size

        def __getitem__(self, idx):
            """Returns tuple (input, target) correspond to batch #idx."""
            i = idx * self.batch_size
            batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
            batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
            x = np.zeros((batch_size,) + self.img_size + (3,), dtype="float32")
            for j, path in enumerate(batch_input_img_paths):
                img = load_img(path, target_size=self.img_size)
                x[j] = img
            y = np.zeros((batch_size,) + self.img_size + (1,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                y[j] = np.expand_dims(img, 2)
            return x, y

    # # 定義 U-Net 模型
    # def unet_model(img_size, num_classes):
    #     inputs = tf.keras.layers.Input(shape=img_size+(3,))
    #     conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    #     conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    #     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    #     conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    #     conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    #     pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    #     conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    #     conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    #     pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    #     conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    #     conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    #     # drop4 = layers.Dropout(0.5)(conv4)
    #     # pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    #     # conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    #     # conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    #     # drop5 = layers.Dropout(0.5)(conv5)

    #     # up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop4)
    #     # merge6 = layers.concatenate([drop4, up6], axis=3)
    #     # conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    #     # conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    #     up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    #     merge7 = layers.concatenate([conv3, up7], axis=3)
    #     conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    #     conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    #     up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    #     merge8 = layers.concatenate([conv2, up8], axis=3)
    #     conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    #     conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    #     up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    #     merge9 = layers.concatenate([conv1, up9], axis=3)
    #     conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    #     conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    #     conv9 = layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
    #     output = layers.Conv2D(3, 1, activation='sigmoid', padding='same')(conv9)
    #     output =tf.keras.layers.Dense(num_classes, activation='softmax')(output)
    #     model = tf.keras.Model(inputs=inputs, outputs=output)
   
    #     return model
    def unet_model(img_size):
        inputs = tf.keras.Input(shape=img_size + (3,))
    
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        b1 = layers.BatchNormalization()(c1)
        r1 = layers.ReLU()(b1)
        r1=layers.Dropout(0.2)(r1)
        p1 = layers.MaxPooling2D((2, 2))(r1)
    
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        b2 = layers.BatchNormalization()(c2)
        r2 = layers.ReLU()(c2)
        r2=layers.Dropout(0.2)(r2)
        p2 = layers.MaxPooling2D((2, 2))(r2)
    
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
        b3 = layers.BatchNormalization()(c3)
        r3 = layers.ReLU()(c3)
        r3=layers.Dropout(0.2)(r3)
        p3 = layers.MaxPooling2D((2, 2))(r3)
    
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
        #b4 = layers.BatchNormalization()(c4)
        #r4 = layers.ReLU()(b4)
        #p4 = layers.MaxPooling2D((2, 2))(r4)
    
        #c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    
        #u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        #u6 = layers.BatchNormalization()(u6)
        #u6 = layers.ReLU()(u6)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
        u7 = layers.concatenate([u7, c3])
        u7 = layers.BatchNormalization()(u7)
        u7 = layers.ReLU()(u7)


        u8 =layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u7)
        u8 = layers.concatenate([u8, c2])
        u8 = layers.BatchNormalization()(u8)
        u8 = layers.ReLU()(u8)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u8)
        u9 = layers.concatenate([u9, c1])
        u9 = layers.BatchNormalization()(u9)
        u9 = layers.ReLU()(u9)

        outputs = layers.Conv2D(3, 1, padding='same')(u9)
        model = tf.keras.Model(inputs, outputs)
        return model
    def ssim_loss(y_true, y_pred):
        c1=tf.reduce_mean(tf.image.ssim(y_true[0], y_pred[0], 1.0))
        c2=tf.reduce_mean(tf.image.ssim(y_true[1], y_pred[1], 1.0))
        c3=tf.reduce_mean(tf.image.ssim(y_true[2], y_pred[2], 1.0))
        return 1-(c1+c2+c3)/3
    
    val_samples = 100
    random.Random(2103).shuffle(input_img_paths)
    random.Random(2103).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)


    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()
    model = unet_model(img_size)
    model.summary()
    model.compile(optimizer="sgd", loss="MSE",metrics=['accuracy'])
    # model.compile(optimizer="adam", loss=ssim_loss, metrics=['accuracy'])
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]
    epochs=100
    
    model.fit(train_gen, epochs=epochs, validation_data=val_gen,callbacks=callbacks)
    # history = model.fit(
    # train_generator,
    # epochs=epochs,
    # validation_data=validation_generator)


    loss = model.history.history['loss']
    acc=model.history.history['accuracy']

    plt.figure(figsize=(16,8))
    plt.subplot(1,4,1)
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.legend()
    plt.subplot(1,4,2)
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.subplot(1,4,3)
    plt.plot(model.history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.subplot(1,4,4)
    plt.plot(model.history.history['val_accuracy'], label='Validation accuracy')
    #plt.plot(model.history.history['mean_square_error'], label='MSE')
    plt.legend()
    plt.show()

    img_size = (360, 640)
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths) 
    val_preds = model.predict(val_gen)

    def displayRes(i):
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_preds[i]))
        return img
    a=[10,20,30]
    plt.figure(figsize=(16,8))
    for idx,i in enumerate(a):  
        plt.subplot(3,3,idx*3+1)
        plt.title("origin")
        x=cv2.imread(val_input_img_paths[i])
        x=cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        plt.imshow(x)
        plt.subplot(3,3,idx*3+2)
        plt.title("predict")
        plt.imshow(displayRes(i))
        plt.subplot(3,3,idx*3+3)
        plt.title("answer")
        x=cv2.imread(val_target_img_paths[i])
        x=cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        plt.imshow(x)
    plt.tight_layout() 
    plt.show()
  