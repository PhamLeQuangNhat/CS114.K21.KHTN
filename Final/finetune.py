from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

from dataloader.simpledatasetloader import SimpleDatasetLoader

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model

from transfer.fcheadnet import FCHeadNet

import argparse
from imutils import paths
import numpy as np
import os

# construct the argument parser and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-m", "--model", required=True,
                help="path to output model")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

    # grab the list of images 
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args['dataset']))

    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]

    # initialize the image preprocessors
    sp = SimplePreprocessor(224,224)
    iap = ImageToArrayPreprocessor()

    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0

    # partition the data into training:75% and testing:25% 
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                    test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # load the VGG16 network, ensuring the head FC layer 
    # sets are left off
    baseModel = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

    # initialize the new head of the network, a set of FC layers
    # followed by a softmax classifier
    headModel = FCHeadNet.build(baseModel, len(classNames), 256)

    # place the head FC model on top of the base model,
    # become the actual model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they
    # will not be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable)
    print("[INFO] compiling model...")
    opt = RMSprop(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

    # train the head of the network for a few epochs (all other
    # layers are frozen) 
    print("[INFO] training head...")
    #model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
    #                    validation_data=(testX, testY), epochs=20,
    #                    steps_per_epoch=len(trainX) // 32, verbose=1)
    H = model.fit(trainX, trainY, validation_data=(testX,testY),
                batch_size=32, epochs=20, verbose=1)

    # evaluate the network after initialization
    print("[INFO] evaluating after initialization...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1), target_names=classNames))

    # unfreeze the final set of CONV layers and make them trainable
    for layer in baseModel.layers[15:]:
        layer.trainable = True

    #recompile the model
    print("[INFO] re-compiling model...")
    opt = SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

    # train the model again
    print("[INFO] fine-tuning model...")
    #model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
    #                validation_data=(testX, testY), epochs=30,
    #                steps_per_epoch=len(trainX) // 32, verbose=1)
    H = model.fit(trainX, trainY, validation_data=(testX,testY),
                batch_size=32, epochs=32, verbose=1)

    # save the network to disk 
    print("[INFO] serializing network ...")
    model.save(args["model"])

    # evaluate the network
    print("[INFO] evaluating after fine-tuning...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=classNames))

if __name__== '__main__':
    main()




