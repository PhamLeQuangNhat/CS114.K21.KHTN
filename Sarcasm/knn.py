from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd 
import re
import string

from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer

# load data
data = pd.read_json("Dataset/SarcasmDataset.json",lines=True)
data = data.drop(columns='article_link')
#print(len(data['headline']))

# turn a doc into clean tokens
def clean_text(text):

    # split inti tokens by while spaces
    #tokens = doc.split()
    text = text.lower()

    # remove text in square brackets
    text = re.sub('\[.*?\]','',text)

    # remove punctuation
    text= re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    # Get rid of some additional punctuation and non-sensical text
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)

    # split inti tokens by while spaces
    tokens = text.split()
    
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    
    return tokens

vocab = []
# load all docs in a directory
def process_texts(data, vocab):
    for text in data['headline']:
        vocab.append(clean_text(text))
    return vocab

vocab = process_texts(data,vocab)

#print(len(vocab))

def crete_tokenizer(dataset_train):
    tokenizer =  Tokenizer()
    tokenizer.fit_on_texts(dataset_train)
    return tokenizer

trainX, testX, trainY, testY = train_test_split(vocab, data['is_sarcastic'], test_size=0.25, random_state=42)
print(trainX[0])
tokenizer = crete_tokenizer(trainX)
Xtrain = tokenizer.texts_to_matrix(trainX,mode='count')
Xtest  = tokenizer.texts_to_matrix(testX,mode='count')
"""
temp = 0
for i in Xtrain[0]:
    if i != 0:
        temp +=1
print(temp)

print(Xtrain.shape, Xtest.shape)

"""
# train the k_NN
print("[INFO] training k_NN ...")
model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
model.fit(Xtrain, trainY)

# evaluate a k-NN 
print("[INFO] evaluating K_NN...")
predictions = model.predict(Xtest)
print(classification_report(testY.argmax(axis=1),
     predictions.argmax(axis=1),
     target_names=classNames))