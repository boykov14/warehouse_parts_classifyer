import os
import numpy as np
import tensorflow as tf
import pandas as pd

from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import  Nadam
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#getting necessary values from local files
# from warehouse_parts_classifyer.data_processing import DataGenerator, process_data, get_detector_mask, get_anchors, get_classes
from warehouse_parts_classifyer.visualisation import PlotLearning

#getting defaults
from warehouse_parts_classifyer.parameters.default_values import DATAPATH, RESOLUTION, RESTORE_PATHS

#setting up plotting class
plot_losses = PlotLearning()

#defining class which we will use for creating/manipulating model
class Parts_Classifier():

    def __init__(self, n_classes, batch_size,  lr = 0.0001, dc = 0.004, save_weights = 0):
        self.resolution = RESOLUTION
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.train_datagen = ImageDataGenerator(
            rotation_range=40,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = self.train_datagen.flow_from_directory(
            'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\train_processed',
            target_size=(RESOLUTION[0], RESOLUTION[1]),
            batch_size=batch_size,
            class_mode='categorical')
        self.validation_generator = self.test_datagen.flow_from_directory(
            'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\test_processed',
            target_size=(RESOLUTION[0], RESOLUTION[1]),
            batch_size=batch_size,
            class_mode='categorical')

        optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=dc)
        self.model = self.create_classifier()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if save_weights:
            self.model.load_weights(RESTORE_PATHS[0])

        # self.training_generator, self.validation_generator = self.extract_data(DATAPATH)

    def create_classifier(self):

        input = Input(shape = (RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))

        X = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(input)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)
        X = MaxPooling2D()(X)

        X = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)

        X = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)

        X = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)
        X = MaxPooling2D(name='middle_layer')(X)

        X = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)

        X = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)

        X = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.1)(X)
        X = MaxPooling2D()(X)

        X = Flatten()(X)
        X = Dense(128, activation='relu')(X)
        X = Dropout(0.5)(X)

        X = Dense(self.n_classes, activation='softmax')(X)

        return Model(input, X)


    def train_model(self):


        checkpoint = ModelCheckpoint(RESTORE_PATHS[0], monitor='val_loss', save_weights_only=True, save_best_only=False)
        checkpoint_best = ModelCheckpoint(RESTORE_PATHS[1], monitor='val_loss', save_weights_only=True, save_best_only=True)

        self.model.fit_generator(
            self.train_generator,
            epochs=1000,
            steps_per_epoch= 50,
            validation_data=self.validation_generator,
            verbose = 1,
            callbacks = [checkpoint, checkpoint_best, plot_losses],
            shuffle = True)

        self.model.save_weights(RESTORE_PATHS[0])

    def test_generater(self):

        img = load_img('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\train_processed\\processed_Part001\\Capture_20180707-164643.jpg')  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in self.train_datagen.flow(x, batch_size=1,
                                  save_to_dir='preview', save_prefix='test', save_format='jpeg'):
            i += 1
            if i > 20:
                break

a = Parts_Classifier(2, 10,  lr = 0.001, dc = 0.001)#, save_weights = RESTORE_PATHS[0])
# a.test_generater()
a.train_model()