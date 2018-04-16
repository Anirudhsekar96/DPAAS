from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

k.set_image_data_format('channels_first')

import os
import importlib


img_width, img_height = 224, 224
#train_data_dir = "/media/anirudh/Data/Code/Dpaas/KERAS_FACE_RECOGNITION/data copy/data copy/train"
train_data_dir = "/media/anirudh/Data/Code/Dpaas/KERAS_FACE_RECOGNITION/i_data_for_training"
#train_data_dir = "/media/anirudh/Data/Code/Dpaas/KERAS_FACE_RECOGNITION/i_video_new"
#validation_data_dir = "/media/anirudh/Data/Code/Dpaas/KERAS_FACE_RECOGNITION/data copy/data copy/val"
validation_data_dir = "/media/anirudh/Data/Code/Dpaas/KERAS_FACE_RECOGNITION/i_data_for_validation"

nb_train_samples = 1000
nb_validation_samples = 200 
batch_size = 16
epochs = 100
weights_path = 'vgg-face-keras-fc.h5'
#weights_path = 'vgg16_face_scraped_personal.h5'
# model = applications.VGG16(weights = "vgg-face-keras-fc.h5", include_top=True, input_shape = (3, img_width, img_height))
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_height, img_width)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Add Fully Connected Layer
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2622, activation='softmax'))
# out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

# model = Model(input=img, output=out)
model.load_weights(weights_path)

# Truncate and replace softmax layer for transfer learning
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(15, activation='softmax'))
"""
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 256, 256, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 256, 256, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 256, 256, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 128, 128, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 128, 128, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 128, 128, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 64, 64, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 64, 64, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 64, 64, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 64, 64, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 64, 64, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 32, 32, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 32, 32, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 32, 32, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 32, 32, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 32, 32, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 16, 16, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
=================================================================
Total params: 20,024,384.0
Trainable params: 20,024,384.0
Non-trainable params: 0.0
"""

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:13]:
    layer.trainable = False

#Adding custom Layers 
# x = model.output
# x = Flatten()(x)
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(1024, activation="relu")(x)
# predictions = Dense(3, activation="softmax")(x)

# creating the final model 
# model = Model(input = model.input, output = out)

# compile the model 
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True,fill_mode = "nearest",zoom_range = 0.3,width_shift_range = 0.3,height_shift_range=0.3,rotation_range=30)

test_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True,fill_mode = "nearest",zoom_range = 0.3,width_shift_range = 0.3,height_shift_range=0.3,rotation_range=30)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size = (img_height, img_width),batch_size = batch_size, class_mode = "categorical", shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size = (img_height, img_width),class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_face_scraped_personal.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

#print(nb_validation_samples/batch_size)
# Train the model 
model.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples/batch_size,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples/batch_size ,
callbacks = [checkpoint,early]
)
