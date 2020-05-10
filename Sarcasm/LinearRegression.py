from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import pandas as pd 


# load data
data = pd.read_json("Dataset/SarcasmDatasetv2.json",lines=True)
X = data['headline']
Y = data['is_sarcastic']

trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.25,random_state=42)
"""
#tfidf= TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(2,2),stop_words='english')
#print(tfidf)
print(len(trainX))

tf = TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_features=25000)  
trainX = tf.fit_transform(trainX.values).toarray()
trainVocab = tf.vocabulary_ 
tf = TfidfVectorizer(vocabulary=trainVocab)
testX = tf.fit_transform(testX.values).toarray()

model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
model.fit(trainX, trainY)
predictions = model.predict(testX)
print(classification_report(testY,predictions))



print(len(trainX[0]))
print(len(testX[1]))
print(len(testX[0]))
"""


for j in range(1):
    if j == 0:
        print("[INFO] Used CounterVectorizer ... ")
        cv = CountVectorizer(stop_words='english')

        trainX = cv.fit_transform(trainX.values).toarray()
        trainVocab = cv.vocabulary_ 
        cv = CountVectorizer(vocabulary=trainVocab)
        testX = cv.fit_transform(testX.values).toarray()
    
    if j == 1:
        print("[INFO] Used TfidfVectorizer ... ")
        tf = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
        
        trainX = tf.fit_transform(trainX.values).toarray()
        trainVocab = tf.vocabulary_ 
        tf = CountVectorizer(vocabulary=trainVocab)
        testX = tf.fit_transform(testX.values).toarray()

    model = LinearRegression()
    print(" [INFO] evaluating Linear Regression...")
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    print(classification_report(testY,predictions))
"""
        if i == 1:
            model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
            print(" [INFO] evaluating K_NN...")
        if i == 2:
            model = LinearRegression()
            print(" [INFO] evaluating Linear Regression...")
        if i == 3:
            model = SVC(kernel='linear')
            print(" [INFO] evaluating SVM...")

        model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        model.fit(trainX, trainY)
        predictions = model.predict(testX)
        print(classification_report(testY,predictions))

"""

