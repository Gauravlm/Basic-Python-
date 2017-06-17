# Kaggle.com
# Bag of words

import pandas as pd
import numpy as np
from textblob import TextBlob

train = pd.read_csv('D:\Study\DS\Kaggle_Project\Bag of Words Meets Bags of Popcorn\labeledTrainData.tsv',
                    header=0,
                    delimiter= '\t',
                    quoting= 3 )

#quoting=3 tells Python to ignore doubled quotes

train.shape
train.columns.values

print(train.review[0])

# in review we found html tags <br> to remove we are using BeautifulSoup API
from bs4 import BeautifulSoup
example1 = BeautifulSoup(train.review[0])
print(train.review[0])
print(example1.get_text())


import re
letters_only= re.sub("[^a-zA-Z]",           # pattern to search for
                     " ",                   # pattern to replace it with
                     example1.get_text())   # text to search
# find anything that is not a lower case(a-z) and upper case(A-Z) and replcae with space

print(letters_only)

# convert lower case
lower_case = letters_only.lower()
# split into the words
words= lower_case.split()

# there are lots of is,an,the,a to removie we are using nltk library
from nltk.corpus import stopwords 

# remove the words from word
words =[w for w in words if not w in stopwords.words("english")]
print(words)

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters  
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 

# get the number of reviews
num_reviews=train["review"].size
     
clean_train_reviews=[]

# cleaning the 1000 reviews

for i in range(0,num_reviews):
    clean_train_reviews.append(review_to_words(train['review'][i]))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer= "word",
                             tokenizer= None,
                             preprocessor= None,
                             stop_words= None,
                             max_features= 5000)


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.

train_data_features= vectorizer.fit_transform(clean_train_reviews)

train_data_features= train_data_features.toarray()  # 25000 rows 5000 features

# take a look at word at vocabulary                                                
vocab= vectorizer.get_feature_names()                                                
print(vocab)

# count the each word in vocabulary
dist = np.sum(train_data_features,axis=0)

for tag,count in zip(vocab,dist):
    print(count,tag)

# Use Random forest with 100 trees

from sklearn.ensemble import RandomForestClassifier
forest= RandomForestClassifier(n_estimators=100)
forest.fit(train_data_features,train['sentiment'])

# read the test data
test = pd.read_csv('D:\Study\DS\Kaggle_Project\Bag of Words Meets Bags of Popcorn\\testData.tsv',
                   header=0,
                   delimiter='\t',
                   quoting=3)
# quoting=3 for ignore double quote

test.columns.values
test.shape
# we need to predict sentiment for test data set
num_review = len(test['review'])
clean_test_review=[]  # creating empty list

print("Cleaing and pasring the test movie review....")
for i in range(0,num_review):
    clean_test_review.append(review_to_words(test['review'][i]))

test_data_features= vectorizer.transform(clean_test_review)
test_data_features= test_data_features.toarray()

# use random forest to predict sentiment label prediction
prediction= forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column

output = pd.DataFrame(data={"id":test['id'],'sentiment':prediction})
