import tensorflow as tf
import numpy as np
import sys

from preprocess import Word2Vec, MSRP, WikiQA, ComplexSimple
from ABDNN import ABDNN
from utils import build_path
from sklearn import linear_model, svm
from sklearn.externals import joblib
import os
import pickle

def train_deconv(lr, w, L2_reg, epoch, batch_size, num_layers)

model = ABDNN(s=train_data.max_len, w=w, l2_reg=l2_reg, model_type='',
                  num_features=train_data.num_features, num_classes=num_classes, num_layers=num_layers)


    optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    saver = tf.train.Saver(max_to_keep=100)