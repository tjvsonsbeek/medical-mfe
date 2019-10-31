from keras.models import Model
from keras.layers import Input, Conv2D,MaxPooling2D, UpSampling2D, concatenate, Dropout,add, Dense
from python_metal_fe.utils import model_tune_generator, historyPlot, dice_coef_loss, auc, mean_iou, dice_coef
import os
from tqdm import tqdm
import numpy as np
import cv2
import random
import copy

class EncoderDecoderNetwork():
    def __init__(self, name, id = 0, weights_path = '/tuned_models'):
        self.name = name
        self.id = id
        self.callbacks = []
        self.weights_path = weights_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)

    def load_weights(self):
        self.weights_file = os.path.join(self.weights_path,"model_{}_{}.h5".format(self.task, self.name))
        self.model.load_weights(self.weights_file)


    def update_encoder_weights(self):
        if self.name == 'VGG16':
            self.feature_extractor = Model(inputs = self.model.input, outputs = self.model.layers[18].output)
        elif self.name == 'ResNet50':
            self.feature_extractor = Model(inputs = self.model.input, outputs = self.model.layers[172].output)
        elif self.name == 'MobileNetV1':
            self.feature_extractor = Model(inputs = self.model.input, outputs = self.model.layers[81].output)
        else:
            raise AssertionError("No weights to update!!")

    def add_callback(self, callback):
        self.callbacks.append(callback)
    def train(self, train_data, val_data, imageDimensions, verbosity):
        self.history = self.model.fit_generator(fake_tune_generator(train_data, self.minibatch_size, imageDimensions), steps_per_epoch = 200, nb_epoch = self.epochs, validation_data =model_tune_generator(val_data, self.minibatch_size, imageDimensions), validation_steps = 50, verbose = verbosity)
    def save_model(self):
        print("---SAVING MODEL---")
        self.model.save_weights(os.path.join(self.weights,"model_{}_{}.h5".format(self.task, self.name)))
    def build_encoder(self):
        if self.name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            self.feature_extractor = VGG16(weights='imagenet', include_top=False)
        elif self.name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            self.feature_extractor = VGG19(weights='imagenet', include_top=False)
        elif self.name == 'ResNet50':
            from keras.applications.resnet50 import ResNet50
            self.feature_extractor = ResNet50(input_shape = (224,224,3),weights='imagenet', include_top=False)
        elif self.name == 'MobileNetV1':
            from keras.applications.mobilenet import MobileNet
            self.feature_extractor = MobileNet(input_shape = (224,224,3), weights='imagenet', include_top=False)
        elif self.name == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            self.feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
        else:
            raise AssertionError("FAILURE!! No Encoder found")

    def build_classifier(self):

        if self.name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            self.classifier = VGG16(weights='imagenet', include_top=False)
        elif self.name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            self.classifier = VGG19(weights='imagenet', include_top=False)
        elif self.name == 'ResNet50':
            from keras.applications.resnet50 import ResNet50
            self.classifier = ResNet50(weights='imagenet', include_top=False)
        elif self.name == 'MobileNetV1':
            from keras.applications.mobilenet import MobileNet
            self.classifier = MobileNet(input_shape = (224,224,3), weights='imagenet', include_top=False)
        else:
            raise AssertionError("FAILURE!! No classifier found")
        # build top:
        dense1 = Dense(4096, activation = 'relu')(self.classifier.layers[-1].output)
        dense2 = Dense(1000, activation = 'relu')(dense1)
        dense3 = Dense(9, activation = 'softmax')(dense2)
        self.classifier = Model(inputs = self.classifier.layers[0].output, outputs = dense3)

    def build_decoder(self):
        if not hasattr(self, 'feature_extractor'):
            raise AttributeError('No feature extractor loaded yet')
        if self.name == 'VGG16':
            self.build_decoder_VGG16()
        elif self.name == 'VGG19':
            self.build_decoder_VGG19()
        elif self.name == 'ResNet50':
            self.build_decoder_RESNET50()
        elif self.name == 'MobileNetV1':
            self.build_decoder_MobileNetV1()
        else:
            raise AssertionError("FAILURE!! No Decoder found")
    def build_decoder_VGG16(self):
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.feature_extractor.layers[18].output)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([self.feature_extractor.layers[17].output,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[13].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([self.feature_extractor.layers[9].output,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[5].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[2].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model =  Model(inputs = self.feature_extractor.layers[0].output, outputs = conv10)
    def build_decoder_VGG19(self):
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.feature_extractor.layers[21].output)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([self.feature_extractor.layers[20].output,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[15].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([self.feature_extractor.layers[10].output,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[5].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[2].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model =  Model(inputs = self.feature_extractor.layers[0].output, outputs = conv10)
    def build_decoder_RESNET50(self):
        conv5 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.feature_extractor.layers[172].output)
        up6 = Conv2D(2048, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([self.feature_extractor.layers[140].output,up6], axis = 3)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[78].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([up8,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[3].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[0].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model = Model(inputs = self.feature_extractor.layers[0].output, outputs = conv10)

    def build_decoder_MobileNetV1(self):
        up6 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(self.feature_extractor.layers[81].output))
        merge6 = concatenate([self.feature_extractor.layers[69].output,up6], axis = 3)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[33].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([self.feature_extractor.layers[21].output,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[9].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[0].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model = Model(inputs = self.feature_extractor.layers[0].output, outputs = conv10)
