# -*- coding: UTF-8 -*-
from tensorflow.keras import layers, models, Model, Sequential

def AlexNet_v1(im_height=224, im_width=224, class_num=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    x = layers.ZeroPadding2D(((1,2),(1,2)))(input_image) # 补零
    x = layers.Conv2D(48,kernel_size=11,strides=4,activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=3,strides=2)(x)
    x = layers.Conv2D(128,kernel_size=5,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3,strides=2)(x)
    x = layers.Conv2D(192,kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(192,kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128,kernel_size=3,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3,strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dense(class_num)(x)

    predict = layers.Softmax()(x)
    model = models.Model(inputs=input_image,outputs=predict)
    return model