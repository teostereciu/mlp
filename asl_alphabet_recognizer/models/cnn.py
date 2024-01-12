import os

import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

from asl_alphabet_recognizer.data.preprocessing import DataHandler
from config.settings import data_dir, models_dir, RANDOM_SEED

from keras.utils import to_categorical

NUM_CLASSES = 24
INPUT_SHAPE = (200, 200, 1)
MODEL_PATH = os.path.join(models_dir, 'final_model.h5')

data_handler = DataHandler(data_dir)
(data_handler.unzip()
 .create_filename_dataframe()
 .sample(n=200)
 .process())

df_train, df_test = data_handler.get_dfs()
X_train = np.array(df_train['X'].tolist())
y_train = np.array(df_train['category'].tolist())
X_test = np.array(df_test['X'].tolist())
y_test = np.array(df_test['category'].tolist())

y_train_onehot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

# build cnn model
model = Sequential()

''' conv2d_12 (Conv2D)          (None, 200, 200, 32)      896       
 conv2d_13 (Conv2D)          (None, 200, 200, 32)      9248                                 
 max_pooling2d_6 (MaxPooling  (None, 100, 100, 32)     0         
 2D)                                                             
 dropout_10 (Dropout)        (None, 100, 100, 32)      0                             
 conv2d_14 (Conv2D)          (None, 100, 100, 64)      18496     
 conv2d_15 (Conv2D)          (None, 100, 100, 64)      36928     
 max_pooling2d_7 (MaxPooling  (None, 50, 50, 64)       0         
 2D)                                                             
 dropout_11 (Dropout)        (None, 50, 50, 64)        0         
 conv2d_16 (Conv2D)          (None, 50, 50, 128)       73856     
 conv2d_17 (Conv2D)          (None, 50, 50, 128)       147584    
 dropout_12 (Dropout)        (None, 50, 50, 128)       0         
 flatten_1 (Flatten)         (None, 320000)            0         
 dense_6 (Dense)             (None, 512)               163840512  
 dropout_13 (Dropout)        (None, 512)               0        
 dense_7 (Dense)             (None, 128)               65664     
 dropout_14 (Dropout)        (None, 128)               0                                                                
 dense_8 (Dense)             (None, 29)                3741'''

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE, strides=(5, 5)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train

mc = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=2)

X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=RANDOM_SEED)

history = model.fit(X_train,
                    y_train_onehot,
                    batch_size=256,
                    epochs=50,
                    verbose=1,
                    validation_data=(X_val, y_val_onehot),
                    callbacks=[mc, es])

