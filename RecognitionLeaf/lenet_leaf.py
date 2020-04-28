from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

from dataloader.simpledatasetloader import SimpleDatasetLoader

from nn.conv.lenet import LeNet

from keras.optimizers import SGD
from imutils import paths
import os
import numpy as np 

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images('Dataset Leaf'))

classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data, labels) = sdl.load(imagePaths, verbose=100)
data = data.astype("float")/255.0

# partition the data into training:75% and testing:25% 
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                               test_size=0.25, random_state=42)

# convert the labels from intergers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=32)
model.compile(loss="categorical_crossentropy", optimizer=opt,
                    metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
             batch_size=32, epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=classNames))

