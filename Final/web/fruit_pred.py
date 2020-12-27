import cv2
import imutils
import joblib
from skimage import feature
import numpy as np
def preprocess(image, width, height):
     
        # grab the dimesions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if the width is smller than the height,so resize
        # along the width and then update the deltas 
        # to crop the height to the desired demension
        if w < h:
            image =imutils.resize(image, width=width, inter=cv2.INTER_AREA)
            dH = int((image.shape[0] - height) / 2.0)

        # otherwise, the height is smaller than the width so 
        # resize along the height and then update the deltas 
        # to crop along the width
        else:
            image = imutils.resize(image, height=height, inter=cv2.INTER_AREA)
            dW = int((image.shape[1] - width) / 2.0)

        # now that our images have been resized, we need to 
        # re-grab the width and height, followed by performing
        # the crop
        (h,w) = image.shape[:2]
        image = image[dH:h -dH, dW:w-dW]

        # finally, resize the image to the provided spatial
        # demensions to ensure ourput image is slways a fixed
        # size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #hg = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
	#			        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        #return hg
        lbp = feature.local_binary_pattern(gray, 24,8, method="uniform")
        
        (hist, _) = np.histogram(lbp.ravel(),
          bins=np.arange(0, 24 + 3),
          range=(0, 24 + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        # return the histogram of Local Binary Patterns
        return hist

def predict():
    class_name = ['Apple','Avocado','Banana','Coconut','Custard_apple',
                'Dragon_fruit','Guava','Mango','Orange','Plum',
                'Start_fruit','Watermelon']
    model_path = 'model/knn_lbps.sav'
    image_path = 'static/image/new_image'
    model = joblib.load(model_path)

    image = cv2.imread(image_path)
    image = preprocess(image,32,32)
    image = image.astype("float") 
    image = image.reshape((1,26))
    preds = model.predict(image)[0]
    #print(preds)
    #print(image.shape) 
    # return class_name[preds]
    return preds

