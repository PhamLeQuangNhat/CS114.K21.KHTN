from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import pandas as pd 


# load data
data = pd.read_json("Dataset/SarcasmDatasetv2.json",lines=True)
X = data['headline']
Y = data['is_sarcastic']

# split test:25%, train:75%
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.25,random_state=42)

# initialize TfidfVectorizer
tf = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)

# transform words to vetor
trainX = tf.fit_transform(trainX.values).toarray()
trainVocab = tf.vocabulary_ 
tf = TfidfVectorizer(vocabulary=trainVocab)
testX = tf.fit_transform(testX.values).toarray()
print("[INFO] Used TfidfVectorizer ... ")

for i in range(3):
    
    if i == 0:
        model = LogisticRegression()
        print("   [INFO] evaluating Logistic Regression...")

    if i == 2:
        model = LinearSVC()
        print("   [INFO] evaluating SVM...")
    
    if i == 1:
        model = MultinomialNB()
        print("   [INFO] evaluating Naive Bayes...")
    # train and evaluating 
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    print(classification_report(testY,predictions))
    


