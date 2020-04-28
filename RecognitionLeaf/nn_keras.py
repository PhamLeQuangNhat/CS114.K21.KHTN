from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from dataloader.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor

from imutils import paths
import os
import numpy as np 


# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images('Dataset Leaf'))
#print(imagePaths)

classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessor, load the data set from disk
# and reshape the data matrix
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=100)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# define the 1024-512-128-32 architecture using keras
model = Sequential()
model.add(Dense(512, input_shape=(1024,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(32, activation="softmax"))

# train the model using SGD
print("[INFO] training network ...")
sgd = SGD(0,01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
             metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
            epochs=100, batch_size=100)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=100)
print(classification_report(testY.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()


