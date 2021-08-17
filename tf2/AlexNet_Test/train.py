# -*- coding: UTF-8 -*-
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1
import tensorflow as tf
import json
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)
        exit(-1)

data_root = os.getcwd().replace('\\','/')
train_dir = data_root+'/train'
validation_dir = data_root+'/val'

im_height = 224
im_width = 224
batch_size = 32
epoch = 10

train_image_generator = ImageDataGenerator(rescale=1./255,
                                 horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./255,)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                target_size=(im_height, im_width),
                                                class_mode='categorical')#分类模型
total_train = train_data_gen.n # 训练样本个数

class_indices = train_data_gen.class_indices
# 索引和值对应
inverse_dict = dict((val,key) for key,val in class_indices.items())
json_str = json.dumps(inverse_dict,indent=4)
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              target_size=(im_height,im_width),
                                                              class_mode='categorical')
total_val = val_data_gen.n # 验证图像数目

# sample_training_images, sample_training_labels = next(train_data_gen)
model = AlexNet_v1(im_height, im_width,class_num=5)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss')]# 保存的判断条件
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train//batch_size,
                    epochs=epoch,
                    validation_data=val_data_gen,
                    validation_steps=total_val//batch_size,
                    callbacks=callbacks)
history_dict = history.history
train_loss = history_dict['loss']
train_accuracy = history_dict['accuracy']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_accuracy']

plt.figure()
plt.plot(range(epoch),train_loss,label='train_loss')
plt.plot(range(epoch),val_loss,label='val_loss')
plt.legend()
plt.tight_layout()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure()
plt.plot(range(epoch),train_accuracy,label='train_accuracy')
plt.plot(range(epoch),val_accuracy,label='val_accuracy')
plt.legend()
plt.tight_layout()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()