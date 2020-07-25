# import thư viện cần thiết
from preprocessing.simplepreprocessor import SimplePreprocessor
from dataloader.simpledatasetloader import SimpleDatasetLoader
from Feature_Extraction.localbinarypatterns import LocalBinaryPatterns
from Feature_Extraction.hog import HOG
from imutils import paths
import os 
import cv2
import argparse
import h5py
import numpy as np 
from sklearn.preprocessing import LabelEncoder

def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("f", "--features", required=True,
                help="path to output save data feature")
    ap.add_argument("-l", "--labels", required=True,
                help="path to output save labels feature")
    ap.add_argument("-t", "--type", required=True,
                help="type feature extraction")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # grab list of images 
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    if args["type"] == "normal":
        sp = SimplePreprocessor(32,32)
        sdl = SimpleDatasetLoader(preprocessors=[sp])
        (data, labels) = sdl.load(imagePaths, verbose=100)
        data = data.astype("float") / 255.0
        data = data.reshape((data.shape[0], 3072))
    elif args["type"] == "lbps":
        sp = SimplePreprocessor(32,32)
        lbps = LocalBinaryPatterns(24,8)
        sdl = SimpleDatasetLoader(preprocessors=[sp,lbps])
        (data, labels) = sdl.load(imagePaths, verbose=100)
        data = data.reshape((data.shape[0], 1024))
    else args["type"] == "HOG":
        sp = SimplePreprocessor(32,32)

    # Encoder labels
    labels = LabelEncoder().fit_transform(labels)

    # save data features
    h5f_data = h5py.File(agrs["features"], 'w')
    h5f_data.create_dataset('dataset', data=np.array(data))

    # save labels
    h5f_label = h5py.File(agrs["labels"], 'w')
    h5f_label.create_dataset('dataset', data=np.array(label))

    h5f_data.close()
    h5f_label.close()

if __name__== '__main__':
    main()