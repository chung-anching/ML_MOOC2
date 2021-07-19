# -*- coding: utf-8 -*-
import numpy as np

try:
    x = np.load("train_x_aug.npy")
    y = np.load("train_y_aug.npy")
except: 
    import dataset as ds
    x,y = ds.load_data_augmentation()
    
x /= 255

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
print(class_weights)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y, 4)

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(64, 64, 3), 
                 activation='relu', padding='same'))
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten()) 

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x, y, class_weight=class_weights, shuffle=True, batch_size=128, epochs=40, validation_split=0.3, verbose=1)

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
show_train_history(history)

# Save model
try:
    model.save_weights("model1.h5")
    print("success")
except:
    print("error")