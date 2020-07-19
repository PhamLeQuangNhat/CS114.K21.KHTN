from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K 

# INPUT => CONV => RELU => FC
class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape
        # to be "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if using "channels first",update the inputShape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, heght, width)
        
        #  define the first (only) CONV = > RELU layer
        model.add(Conv2D(32, (3, 3), padding="same",
                input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
