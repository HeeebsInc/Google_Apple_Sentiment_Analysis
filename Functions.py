import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import enchant
from pprint import pprint
import pickle


def encode_emotion_3(x): 
    x = x.lower() 
    if x == 'negative emotion': 
        return 0 
    elif x == 'no emotion toward brand or product': 
        return 2
    elif x == 'positive emotion': 
        return 1
    else: 
        return None
    
    
def encode_emotion_2(x): 
    x = x.lower() 
    if x == 'negative emotion': 
        return 0 
    elif x == 'positive emotion': 
        return 1
    else: 
        return None
    

# new_df['Emotion_New'] = new_df.Emotion.map(encode_emotion)

def clean_split(split_type, df): 
    new_df = pd.DataFrame() 
    new_df['Text'] = df['tweet_text']
    new_df['Item'] = df['emotion_in_tweet_is_directed_at']
    new_df['Emotion'] = df['is_there_an_emotion_directed_at_a_brand_or_product']
    if split_type == 2: 
        new_df['Emotion_New'] = new_df.Emotion.map(encode_emotion_3)
    else: 
        new_df['Emotion_New'] = new_df.Emotion.map(encode_emotion_2)
    
    #dropping na in columns Text and Emotion
    new_df.dropna(subset = ['Text', 'Emotion_New'], inplace = True)
    
    #getting rid of @ symbols
    en_us = enchant.Dict("en_US")

    phrases = new_df.Text.values

    for i, phrase in enumerate(new_df.Text):
        phrases[i] = ' '.join(w for w in phrase.split() if en_us.check(w))

    new_df.Text = phrases
    
    word_tokenizer = RegexpTokenizer(r'\w+')
    tweet_token = TweetTokenizer()
    new_df.Text = new_df.Text.map(lambda x: tweet_token.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join(x))
    new_df.Text= new_df.Text.map(lambda x: word_tokenizer.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join(x))
    
    
    #split into test and trains
    x_train, x_test, y_train, y_test = train_test_split(new_df.Text, new_df.Emotion_New, stratify = new_df.Emotion_New,                                        train_size = .75, random_state = 10)
    
    #removing stop words
    stop = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words = stop, max_features = 5000, ngram_range=(1,3))
    clean_train = x_train.values
    clean_test = x_test.values

    train_features =vectorizer.fit_transform(clean_train).toarray()
    test_features = vectorizer.fit_transform(clean_test).toarray()
    
    
    
    #pickling
    pickle.dump(train_features, open(f'../Pickles/{split_type}_x_train.p', 'wb'))
    pickle.dump(test_features, open(f'../Pickles/{split_type}_x_test.p', 'wb'))
    pickle.dump(y_train, open(f'../Pickles/{split_type}_y_train.p', 'wb'))
    pickle.dump(y_test, open(f'../Pickles/{split_type}_y_test.p', 'wb'))
    
    print('Finished Pickling')
    
 