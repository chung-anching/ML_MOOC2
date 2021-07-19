# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

img_size = 64

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(img_size, img_size, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.5))
    
    model.add(Flatten()) 

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model

import matplotlib.pyplot as plt

def show_train_history(train_history):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.xticks([row for row in range(0, len(train_history.history['acc']))])
    plt.title('Train History')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.xticks([row for row in range(0, len(train_history.history['loss']))])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

import numpy as np
import dataset as ds
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical

#---------------------------------------------------
try:
    x = np.load("train_x_disease_aug.npy")
    y = np.load("train_y_disease_aug.npy")
except: 
    x,y = ds.load_data_disease_aug()
          
x /= 255
  
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
print(class_weights)
  
y = to_categorical(y, 2)
  
model_disease = build_model()
history = model_disease.fit(x, y, class_weight=class_weights, shuffle=True, batch_size=128, epochs=30, validation_split=0.2, verbose=1)
show_train_history(history)
  
# Save model
try:
    model_disease.save_weights("model_disease_aug.h5")
    print("success")
except:
    print("error")
    
#---------------------------------------------------
try:
    x = np.load("train_x_rust_aug.npy")
    y = np.load("train_y_rust_aug.npy")
except: 
    x,y = ds.load_data_rust_aug()
         
x /= 255
 
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
print(class_weights)
 
y = to_categorical(y, 2)
 
model_rust = build_model()
history = model_rust.fit(x, y, class_weight=class_weights, shuffle=True, batch_size=128, epochs=30, validation_split=0.2, verbose=1)
show_train_history(history)
 
# Save model
try:
    model_rust.save_weights("model_rust_aug.h5")
    print("success")
except:
    print("error")

#---------------------------------------------------
try:
    x = np.load("train_x_scab_aug.npy")
    y = np.load("train_y_scab_aug.npy")
except: 
    x,y = ds.load_data_scab_aug()
         
x /= 255
 
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
print(class_weights)
 
y = to_categorical(y, 2)
 
model_scab = build_model()
history = model_scab.fit(x, y, class_weight=class_weights, shuffle=True, batch_size=128, epochs=30, validation_split=0.2, verbose=1)
show_train_history(history)
 
# Save model
try:
    model_scab.save_weights("model_scab_aug.h5")
    print("success")
except:
    print("error")