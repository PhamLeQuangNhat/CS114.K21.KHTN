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
# print(len(data))
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=42)
#tfidf= TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(2,2),stop_words='english')
#print(tfidf)
for j in range(2):
    if j == 0:
        print("[INFO] Used CounterVectorizer ... ")
        cv = CountVectorizer(stop_words='english')
        trainX = cv.fit_transform(trainX)
        testX  = cv.fit_transform(testX)
    if j == 1:
        print("[INFO] Used TfidfVectorizer ... ")
        tf = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
        trainX = tf.fit_transform(trainX)
        testX  = tf.fit_transform(testX)
    for i in range(4):
        if i == 0:
            model=RandomForestClassifier()
            print(" [INFO] evaluating RandomForest...")
        if i == 1:
            model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
            print(" [INFO] evaluating K_NN...")
        if i == 2:
            model = LinearRegression()
            print(" [INFO] evaluating Linear Regression...")
        if i == 3:
            model = SVC(kernel='linear')
            print(" [INFO] evaluating SVM...")
    	model.fit(trainX, trainY)
    	predictions = model.predict(testX)
    	print(classification_report(testY,predictions))


