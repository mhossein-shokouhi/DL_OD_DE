import os
import cv2
import glob
import argparse
import matplotlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import numpy as np
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

def depth_estimation(pretrained_model = "nyu.h5", input_img = "examples/*.png"):
    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(pretrained_model, custom_objects=custom_objects, compile=False)

    print('\nModel loaded ({0}).'.format(pretrained_model))

    # Input images
    inputs = load_images( glob.glob(input_img) )
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)
    depths = cv2.resize(outputs[0], (0, 0), fx=2, fy=2).reshape((480,640,1))
    np.save('depth.npy', depths)
