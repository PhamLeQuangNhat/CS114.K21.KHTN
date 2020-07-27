"""use python3 lenet_train.py --output image/image_lenet.png 
                      --model H5PY/model/lenet_weights.hdf5""" 

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras import backend as K 
from nn.conv.lenet import LeNet

from keras.optimizers import SGD
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py

# construct the argument parser and parse the arguments
def option():
    ap = argparse.ArgumentParser()               
    ap.add_argument("-o", "--output", required=True,
                help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", required=True,
                help="path to output model")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    class_name = ['Apple','Avocado','Banana','Coconut','Custard_apple',
                  'Dragon_fruit','Guava','Mango','Orange','Plum',
                  'Start_fruit','Watermelon']

    in_data = 'H5PY/train/train_normal.h5'
    in_label = 'H5PY/train/labels_train.h5'

    # import the feature vector and trained labels
    h5f_data  = h5py.File(in_data, 'r')
    h5f_label = h5py.File(in_label, 'r')

    data = h5f_data['dataset']
    labels = h5f_label['dataset']

    data = np.array(data)
    labels = np.array(labels)

    # reshape data matrix
    if K.image_data_format() == "channels_first":
        data = data.reshape(data.shape[0],3,32,32)
    else:
        data = data.reshape(data.shape[0],32,32,3)
    print(data.shape)    
    
    # split training: 80%, testing: 20%
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                    test_size=0.20, random_state=42)
    
  
    # convert labels as vector 
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.05)
    model = LeNet.build(width=32, height=32, depth=3, classes=12)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    
    # train the network 
    print("[INFO]training network ...")
    H = model.fit(trainX, trainY, validation_data=(testX,testY),
                batch_size=32, epochs=40, verbose=1)
    
    # save the network to disk 
    print("[INFO] serializing network ...")
    model.save(args["model"])

    # evaluate the network
    print("[INFO] evaluating network...")
    preds = model.predict(testX)
    print(classification_report(testY.argmax(axis=1),
                    preds.argmax(axis=1),
                    target_names=class_name))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["output"])
    
if __name__== '__main__':
    main()



