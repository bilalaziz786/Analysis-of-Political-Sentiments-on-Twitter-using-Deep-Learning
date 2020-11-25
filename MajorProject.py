# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:47:14 2019

@author: bilal
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:25:35 2018

@author: bilal
"""

import tweepy
import csv
#Twitter API credentials
consumer_key ="xxxxxx"
consumer_secret ="xxxxxxx"
access_key ="xxx-xxx"
access_secret ="xxxx"

def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    #l = StdOutListener()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
   

    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print ("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        except tweepy.TweepError:
            print("Failed to run the command on that user, Skipping...")


        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print ("...%s tweets downloaded so far" % (len(alltweets)))

    #transform the tweepy tweets into a 2D array that will populate the csv
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"),tweet.retweet_count,tweet.favorite_count] for tweet in alltweets]
    #write the csv
    with open('%s_tweets.csv' % screen_name, 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text","retweet_count","favorite_count"])
        writer.writerows(outtweets)



if __name__ == '__main__':
    #pass in the username of the account you want to download
    get_all_tweets("ElectionsnIndia")
    
    
# Natural Language Processing
    
# Importing the libraries

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('New_Political_Tweets.csv', encoding='ISO-8859–1')

# Cleaning the texts
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 3896):
    
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text=dataset['text'][i]
    #'Not' is replaced by 'Nots' so that it will not be detected by stopwords.
    text=re.sub("not","nots",text)
    text = text.lower() 
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

#Comparison models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.sparse import lil_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
# Importing the Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, LSTM
from keras.utils import to_categorical
from keras.optimizers import adam
cv = CountVectorizer(max_features = 500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 2:5]


#Converting it into multiclass classification
ms=[]
for i in range(len(y)):
    count=0
    if y['BJP'][i]==1:
        count+=4
    if y['Congress'][i]==1:
        count+=2
    if y['Others'][i]==1:
        count+=1
    
    ms.append(count)
    
y=pd.DataFrame({'Decimal':ms})
y=y.values.reshape(-1,)
'''
# Note that this classifier can throw up errors when handling sparse matrices.
x_train = lil_matrix(x_train).toarray()
#y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(x_test).toarray()'''

#CountVectorizer
#Take the sum of all accuracies of the 10 folds
sumsvc=0
sumdtc=0
sumrfc=0
sumlr=0
t=0

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x, y)
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
print("SKF on count vectorizer...")
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index] 


    classifiers=[
        (SVC(kernel = 'rbf', random_state = 0),"SVC"),
        (DecisionTreeClassifier(random_state = 0),"DTC"),
        (LogisticRegression(),"LR"),
        (RandomForestClassifier(n_estimators=80, max_depth=100,random_state=0),"RFC"),
    ]
    
    #Accuracy scores of different models
    score_ , names = [] , []
    for model,name in classifiers:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score_.append(accuracy_score(y_test,y_pred)*100)
        names.append(name)
        #print(classification_report(ytest_tfidf, ypred_tfidf))
    
        
    Acv = score_[:]
    sumsvc=sumsvc+Acv[0]
    sumdtc=sumdtc+Acv[1]
    sumlr=sumlr+Acv[2]
    sumrfc=sumrfc+Acv[3]
    #Finding the best split on the basis of RFC
    if Acv[2] > t:
        t=Acv[2]
        xbest_train=x_train
        xbest_test=x_test
        ybest_train=y_train
        ybest_test=y_test
Acv[0]=sumsvc/10
Acv[1]=sumdtc/10
Acv[2]=sumlr/10
Acv[3]=sumrfc/10


#Now let's make the ANN!
y_binary_train = to_categorical(ybest_train)
y_binary_test = to_categorical(ybest_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu', input_dim = 500))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xbest_train, y_binary_train, batch_size = 50, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('ANN')
Acv.append(accuracy_score(y_binary_test,y_pred)*100)



#I prefer not to add drop out in LSTM cells for one specific and clear reason. 
#LSTMs are good for long terms but an important thing about them is that they are not very well at memorising multiple things simultaneously. 
#The logic of drop out is for adding noise to the neurons in order not to be dependent on any specific neuron. 
#By adding drop out for LSTM cells, there is a chance for forgetting something that should not be forgotten. 
#Consequently, like CNNs I always prefer to use drop out in dense layers after the LSTM layers.
#LSTM
xbest_train = np.reshape(xbest_train, (xbest_train.shape[0], 1, xbest_train.shape[1]))
xbest_test = np.reshape(xbest_test, (xbest_test.shape[0], 1, xbest_test.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, 500), activation='relu', return_sequences=True))
#Add a new Dropout layer between the input (or visible layer) and the first hidden layer. 
#The dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from 
#each update cycle.
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='softmax'))

opt = adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(xbest_train,
          ybest_train,
          epochs=100,
          validation_data=(xbest_test, ybest_test))

# Predicting the Test set results
y_pred = model.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
y_binary_test = to_categorical(ybest_test)
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('LSTM')
Acv.append(accuracy_score(y_binary_test,y_pred)*100)


#TFIDF
#Comparison models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

# word level tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect_unigram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,1), max_features=200)
tfidf_vect_unigram.fit(corpus)
x_tfidf_unigram =  tfidf_vect_unigram.transform(corpus).toarray()
#Take the sum of all accuracies of the 10 folds
sumsvc=0
sumdtc=0
sumrfc=0
sumlr=0
t=0

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_tfidf_unigram, y)
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
print("SKF on tfidf (unigram)...")

for train_index, test_index in skf.split(x_tfidf_unigram, y):
    xtrain_tfidf, xtest_tfidf = x_tfidf_unigram[train_index], x_tfidf_unigram[test_index]
    ytrain_tfidf, ytest_tfidf = y[train_index], y[test_index] 


    classifiers=[
        (SVC(kernel = 'rbf', random_state = 0),"SVC"),
        (DecisionTreeClassifier(random_state = 0),"DTC"),
        (LogisticRegression(),"LR"),
        (RandomForestClassifier(n_estimators=80, max_depth=100,random_state=0),"RFC"),
    ]
    
    #Accuracy scores of different models
    score_ , names = [] , []
    for model,name in classifiers:
        model.fit(xtrain_tfidf, ytrain_tfidf)
        ypred = model.predict(xtest_tfidf)
        score_.append(accuracy_score(ytest_tfidf,ypred)*100)
        names.append(name)
        #print(classification_report(ytest_tfidf, ypred_tfidf))
    
        
    
    Atfidf = score_[:]
    sumsvc=sumsvc+Atfidf[0]
    sumdtc=sumdtc+Atfidf[1]
    sumlr=sumlr+Atfidf[2]
    sumrfc=sumrfc+Atfidf[3]
    #Finding the best split on the basis of RFC
    if Atfidf[2] > t:
        t=Atfidf[2]
        xbest_train=xtrain_tfidf
        xbest_test=xtest_tfidf
        ybest_train=ytrain_tfidf
        ybest_test=ytest_tfidf
Atfidf[0]=sumsvc/10
Atfidf[1]=sumdtc/10
Atfidf[2]=sumlr/10
Atfidf[3]=sumrfc/10



#Now let's make the ANN!
y_binary_train = to_categorical(ybest_train)
y_binary_test = to_categorical(ybest_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu', input_dim = 200))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xbest_train, y_binary_train, batch_size = 50, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('ANN')
Atfidf.append(accuracy_score(y_binary_test,y_pred)*100)



#I prefer not to add drop out in LSTM cells for one specific and clear reason. 
#LSTMs are good for long terms but an important thing about them is that they are not very well at memorising multiple things simultaneously. 
#The logic of drop out is for adding noise to the neurons in order not to be dependent on any specific neuron. 
#By adding drop out for LSTM cells, there is a chance for forgetting something that should not be forgotten. 
#Consequently, like CNNs I always prefer to use drop out in dense layers after the LSTM layers.
#LSTM
xbest_train = np.reshape(xbest_train, (xbest_train.shape[0], 1, xbest_train.shape[1]))
xbest_test = np.reshape(xbest_test, (xbest_test.shape[0], 1, xbest_test.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, 200), activation='relu', return_sequences=True))
#Add a new Dropout layer between the input (or visible layer) and the first hidden layer. 
#The dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from 
#each update cycle.
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='softmax'))

opt = adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(xbest_train,
          ybest_train,
          epochs=100,
          validation_data=(xbest_test, ybest_test))

# Predicting the Test set results
y_pred = model.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
y_binary_test = to_categorical(ybest_test)
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('LSTM')
Atfidf.append(accuracy_score(y_binary_test,y_pred)*100)        
        
        
# word level tf-idf (bigrams)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect_bigram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=200)
tfidf_vect_bigram.fit(corpus)
x_tfidf_bigram =  tfidf_vect_bigram.transform(corpus).toarray()
#Take the sum of all accuracies of the 10 folds
sumsvc=0
sumdtc=0
sumrfc=0
sumlr=0
t=0

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_tfidf_bigram, y)
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
print("SKF on tfidf (bigrams)...")

for train_index, test_index in skf.split(x_tfidf_bigram, y):
    xtrain_tfidf, xtest_tfidf = x_tfidf_bigram[train_index], x_tfidf_bigram[test_index]
    ytrain_tfidf, ytest_tfidf = y[train_index], y[test_index] 


    classifiers=[
        (SVC(kernel = 'rbf', random_state = 0),"SVC"),
        (DecisionTreeClassifier(random_state = 0),"DTC"),
        (LogisticRegression(),"LR"),
        (RandomForestClassifier(n_estimators=80, max_depth=100,random_state=0),"RFC"),
    ]
    
    #Accuracy scores of different models
    score_ , names = [] , []
    for model,name in classifiers:
        model.fit(xtrain_tfidf, ytrain_tfidf)
        ypred = model.predict(xtest_tfidf)
        score_.append(accuracy_score(ytest_tfidf,ypred)*100)
        names.append(name)
        #print(classification_report(ytest_tfidf, ypred_tfidf))
    
        
    
    Atfidfbi = score_[:]
    sumsvc=sumsvc+Atfidfbi[0]
    sumdtc=sumdtc+Atfidfbi[1]
    sumlr=sumlr+Atfidfbi[2]
    sumrfc=sumrfc+Atfidfbi[3]
    #Finding the best split on the basis of RFC
    if Atfidfbi[2] > t:
        t=Atfidfbi[2]
        xbest_train=xtrain_tfidf
        xbest_test=xtest_tfidf
        ybest_train=ytrain_tfidf
        ybest_test=ytest_tfidf
Atfidfbi[0]=sumsvc/10
Atfidfbi[1]=sumdtc/10
Atfidfbi[2]=sumlr/10
Atfidfbi[3]=sumrfc/10





#Now let's make the ANN!
y_binary_train = to_categorical(ybest_train)
y_binary_test = to_categorical(ybest_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu', input_dim = 200))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xbest_train, y_binary_train, batch_size = 50, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('ANN')
Atfidfbi.append(accuracy_score(y_binary_test,y_pred)*100)



#I prefer not to add drop out in LSTM cells for one specific and clear reason. 
#LSTMs are good for long terms but an important thing about them is that they are not very well at memorising multiple things simultaneously. 
#The logic of drop out is for adding noise to the neurons in order not to be dependent on any specific neuron. 
#By adding drop out for LSTM cells, there is a chance for forgetting something that should not be forgotten. 
#Consequently, like CNNs I always prefer to use drop out in dense layers after the LSTM layers.
#LSTM
xbest_train = np.reshape(xbest_train, (xbest_train.shape[0], 1, xbest_train.shape[1]))
xbest_test = np.reshape(xbest_test, (xbest_test.shape[0], 1, xbest_test.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, 200), activation='relu', return_sequences=True))
#Add a new Dropout layer between the input (or visible layer) and the first hidden layer. 
#The dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from 
#each update cycle.
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='softmax'))

opt = adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(xbest_train,
          ybest_train,
          epochs=100,
          validation_data=(xbest_test, ybest_test))

# Predicting the Test set results
y_pred = model.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
y_binary_test = to_categorical(ybest_test)
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('LSTM')
Atfidfbi.append(accuracy_score(y_binary_test,y_pred)*100)

        
# word level tf-idf (trigrams)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect_trigram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=200)
tfidf_vect_trigram.fit(corpus)
x_tfidf_trigram =  tfidf_vect_trigram.transform(corpus).toarray()
#Take the sum of all accuracies of the 10 folds
sumsvc=0
sumdtc=0
sumrfc=0
sumlr=0
t=0

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_tfidf_trigram, y)
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
print("SKF on tfidf (trigrams)...")

for train_index, test_index in skf.split(x_tfidf_trigram, y):
    xtrain_tfidf, xtest_tfidf = x_tfidf_trigram[train_index], x_tfidf_trigram[test_index]
    ytrain_tfidf, ytest_tfidf = y[train_index], y[test_index] 


    classifiers=[
        (SVC(kernel = 'rbf', random_state = 0),"SVC"),
        (DecisionTreeClassifier(random_state = 0),"DTC"),
        (LogisticRegression(),"LR"),
        (RandomForestClassifier(n_estimators=80, max_depth=100,random_state=0),"RFC"),
    ]
    
    #Accuracy scores of different models
    score_ , names = [] , []
    for model,name in classifiers:
        model.fit(xtrain_tfidf, ytrain_tfidf)
        ypred = model.predict(xtest_tfidf)
        score_.append(accuracy_score(ytest_tfidf,ypred)*100)
        names.append(name)
        #print(classification_report(ytest_tfidf, ypred_tfidf))
    
        
    
    Atfidftri = score_[:]
    sumsvc=sumsvc+Atfidftri[0]
    sumdtc=sumdtc+Atfidftri[1]
    sumlr=sumlr+Atfidftri[2]
    sumrfc=sumrfc+Atfidftri[3]
    #Finding the best split on the basis of RFC
    if Atfidftri[2] > t:
        t=Atfidftri[2]
        xbest_train=xtrain_tfidf
        xbest_test=xtest_tfidf
        ybest_train=ytrain_tfidf
        ybest_test=ytest_tfidf
Atfidftri[0]=sumsvc/10
Atfidftri[1]=sumdtc/10
Atfidftri[2]=sumlr/10
Atfidftri[3]=sumrfc/10



#Now let's make the ANN!
y_binary_train = to_categorical(ybest_train)
y_binary_test = to_categorical(ybest_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu', input_dim = 200))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xbest_train, y_binary_train, batch_size = 50, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('ANN')
Atfidftri.append(accuracy_score(y_binary_test,y_pred)*100)



#I prefer not to add drop out in LSTM cells for one specific and clear reason. 
#LSTMs are good for long terms but an important thing about them is that they are not very well at memorising multiple things simultaneously. 
#The logic of drop out is for adding noise to the neurons in order not to be dependent on any specific neuron. 
#By adding drop out for LSTM cells, there is a chance for forgetting something that should not be forgotten. 
#Consequently, like CNNs I always prefer to use drop out in dense layers after the LSTM layers.
#LSTM
xbest_train = np.reshape(xbest_train, (xbest_train.shape[0], 1, xbest_train.shape[1]))
xbest_test = np.reshape(xbest_test, (xbest_test.shape[0], 1, xbest_test.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, 200), activation='relu', return_sequences=True))
#Add a new Dropout layer between the input (or visible layer) and the first hidden layer. 
#The dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from 
#each update cycle.
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='softmax'))

opt = adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

history=model.fit(xbest_train,
          ybest_train,
          epochs=100,
          validation_data=(xbest_test, ybest_test))



# Predicting the Test set results
y_pred = model.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
y_binary_test = to_categorical(ybest_test)
print(accuracy_score(y_binary_test,y_pred)*100)
names.append('LSTM')
Atfidftri.append(accuracy_score(y_binary_test,y_pred)*100)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

        
        

#Word2Vec
import gensim 
from nltk import word_tokenize
model = gensim.models.Word2Vec.load("word2vec.model")
Bigger_list=[]
for i in corpus: 
    tokenized_text = word_tokenize(i)
    #print(tokenized_text)
    Bigger_list.append(tokenized_text)
# build vocabulary and train model
model = gensim.models.Word2Vec(
    Bigger_list,
    size=4,
    window=3,
    min_count=5,
    workers=10)
model.train(Bigger_list, total_examples=len(Bigger_list), epochs=20)
print(model)
model.save("word2vec.model")
model.save("model.bin")
vocab = list(model.wv.vocab)
l=model.wv.vectors
df = pd.DataFrame(index=range(3896),columns=range(224))
df[:]=0
for j in range(0, 3896):
    w=0
    text=corpus[j] 
    text = text.split()
    for k in range(0, len(text)):
        if text[k] in vocab and w<=223:
           df.iloc[j,w:w+4]=df.iloc[j,w:w+4] + l[vocab.index(text[k])]
        w=w+4
df.to_csv('out.csv', encoding='utf-8', index=False)
df=df.values
xtrain_word2vec, xtest_word2vec, ytrain_word2vec, ytest_word2vec=train_test_split(df, y, random_state=0)

classifiers=[
    (SVC(kernel = 'rbf', random_state = 0),"SVC"),
    (DecisionTreeClassifier(random_state = 0),"DTC"),
    (LogisticRegression(),"LR"),
    (RandomForestClassifier(n_estimators=80, max_depth=100,random_state=0),"RFC"),
]

#Accuracy scores of different models
score_ , names = [] , []
for model,name in classifiers:
    model.fit(xtrain_word2vec, ytrain_word2vec)
    ypred = model.predict(xtest_word2vec)
    score_.append(accuracy_score(ytest_word2vec,ypred)*100)
    names.append(name)
    #print(classification_report(ytest_tfidf, ypred_tfidf))
    
        
Aw2v = score_[:]   
xbest_train=xtrain_word2vec
xbest_test=xtest_word2vec
ybest_train=ytrain_word2vec
ybest_test=ytest_word2vec


#Now let's make the ANN!
y_binary_train = to_categorical(ybest_train)
y_binary_test = to_categorical(ybest_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu', input_dim = 2800))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xbest_train, y_binary_train, batch_size = 50, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('ANN')
Aw2v.append(accuracy_score(y_binary_test,y_pred)*100)



#I prefer not to add drop out in LSTM cells for one specific and clear reason. 
#LSTMs are good for long terms but an important thing about them is that they are not very well at memorising multiple things simultaneously. 
#The logic of drop out is for adding noise to the neurons in order not to be dependent on any specific neuron. 
#By adding drop out for LSTM cells, there is a chance for forgetting something that should not be forgotten. 
#Consequently, like CNNs I always prefer to use drop out in dense layers after the LSTM layers.
#LSTM
xbest_train = np.reshape(xbest_train, (xbest_train.shape[0], 1, xbest_train.shape[1]))
xbest_test = np.reshape(xbest_test, (xbest_test.shape[0], 1, xbest_test.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, 2800), activation='relu', return_sequences=True))
#Add a new Dropout layer between the input (or visible layer) and the first hidden layer. 
#The dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from 
#each update cycle.
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='softmax'))

opt = adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(xbest_train,
          ybest_train,
          epochs=100,
          validation_data=(xbest_test, ybest_test))

# Predicting the Test set results
y_pred = model.predict(xbest_test)
y_pred = (y_pred > 0.5)

# Computing the accuracy
y_binary_test = to_categorical(ybest_test)
print(accuracy_score(y_binary_test,y_pred)*100)
#print(classification_report(y_binary_test, y_pred))
names.append('LSTM')
Aw2v.append(accuracy_score(y_binary_test,y_pred)*100)


#counting the predictions
countbjp=0
countcongress=0
countothers=0
for i in range(0,391):
    if y_pred[i][4]==True:
        countbjp+=1
    elif y_pred[i][2]==True:
        countcongress+=1
    elif y_pred[i][1]==True:
        countothers+=1

        
#Graph to show the accuracy of different models
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
plt.plot(Acv)
plt.plot(Atfidf)
plt.plot(Atfidfbi)
plt.plot(Atfidftri)
plt.plot(Aw2v)
for i, label in enumerate(names):
    plt.text(i,Acv[i], label) 
    plt.text(i,Atfidf[i], ' ')
    plt.text(i,Atfidfbi[i], ' ')
    plt.text(i,Atfidftri[i], ' ')
    plt.text(i,Aw2v[i], label)
    
plt.legend(['Count Vectorizer', 'TF-IDF (Unigrams)', 'TF-IDF (Bigrams)', 'TF-IDF (Trigrams)', 'Word2Vec'], loc='upper left')
plt.show()
 
# Bar graph to count number of texts per category
df_toxic = dataset.drop(['id', 'text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
'''for i in categories:
    counts.append((i, df_toxic[i].sum()))'''
counts.append(('BJP', countbjp*100/391))
counts.append(('Congress', countcongress*100/391))
counts.append(('Others', countothers*100/391))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats

df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=False, figsize=(8, 5))
plt.title("Predictions")
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Category', fontsize=12)

# Creating the wordcloud
from wordcloud import WordCloud,STOPWORDS
comment_words = ' '

stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in corpus: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '

  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (6, 6), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 
