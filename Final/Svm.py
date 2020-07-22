from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from preprocessing.simplepreprocessor import SimplePreprocessor
from Feature_Extraction.localbinarypatterns import LocalBinaryPatterns
from dataloader.simpledatasetloader import SimpleDatasetLoader

from sklearn.svm import LinearSVC

from imutils import paths
import os
import numpy as np 
import argparse

def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
                help="path to input data")
    ap.add_argument("-f", "--feature", required=True,
                help="choices type of feature extractions")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # grab the list of images 
    print ("[INFO] loading images ...")
    imagePaths = list(paths.list_images(args["dataset"]))
    #print(len(imagePaths))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]

    # initialize the image preprocessors
    sp = SimplePreprocessor(32,32)

    # choices type of features
    if args["feature"] == "simple":
        sdl = SimpleDatasetLoader(preprocessors=[sp])
        (data, labels) = sdl.load(imagePaths, verbose=100)
        data = data.astype("float") / 255.0
        data = data.reshape((data.shape[0], 3072))
    elif args["feature"] == "lbp":
        lbp = LocalBinaryPatterns(24,8)
        sdl = SimpleDatasetLoader(preprocessors=[sp,lbf])
        (data, labels) = sdl.load(imagePaths, verbose=100)
        data = data.astype("float") / 255.0
        data = data.reshape((data.shape[0], 1024))

    # partition the data into training: 75%, testing: 25%
    (trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                    test_size=0.25, random_state=42)

    # convert the labels from intergers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY  = LabelBinarizer().fit_transform(testY)

    # train the k_NN
    print("[INFO] training SVM...")
    model = LinearSVC()
    model.fit(trainX, trainY)

    # evaluate a k-NN 
    print("[INFO] evaluating SVM...")
    predictions = model.predict(testX)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=classNames))

if __name__== '__main__':
    main()
