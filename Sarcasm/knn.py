from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
from collections import Counter


# load data
data = pd.read_json("Dataset/SarcasmDataset.json",lines=True)
X = data['headline']
Y = data['is_sarcastic']
print(len(data))

cv=CountVectorizer(stop_words='english')
X = cv.fit_transform(X)
# print(X[1])

trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=101)
model=RandomForestClassifier()
model.fit(trainX, trainY)
predictions = model.predict(testX)
print(classification_report(testY,predictions))
"""

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.25, random_state=101)
#print(trainX)


# train the k_NN
print("[INFO] training k_NN ...")
#model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
model=RandomForestClassifier()
model.fit(trainX, trainY)

# evaluate a k-NN 
print("[INFO] evaluating K_NN...")
predictions = model.predict(testX)
print(classification_report(testY,predictions))
"""
