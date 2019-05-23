#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:52:10 2019

@author: hayashi
"""
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3


def model_incepv3(n_class,height,width):
    out_num = n_class
    input_tensor = Input(shape=(height,width,3))
    incep = InceptionV3(include_top=False,input_tensor=input_tensor,weights=None)
    x = incep.output
    x = Flatten()(x)
    x = Dense(4096,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(out_num)(x)
    x = Activation("softmax")(x)
    model = Model(inputs=incep.inputs,outputs=x)
    #loss_f = distribution_cross_entropy([40])
    model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])
    return model

def model_vgg(n_class,height,width):
    out_num = n_class
    input_tensor = Input(shape=(height,width,3))
    vgg = VGG16(include_top=False,input_tensor=input_tensor,weights=None)
    x = vgg.output
    x = Flatten()(x)
    x = Dense(2048,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(out_num)(x)
    x = Activation("softmax")(x)
    model = Model(inputs=vgg.inputs,outputs=x)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
