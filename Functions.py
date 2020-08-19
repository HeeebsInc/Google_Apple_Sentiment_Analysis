import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import enchant
import pickle
from sklearn.utils import resample
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import itertools
import seaborn as sns
import re



def get_pickles(split_type): 
    x_train = pickle.load(open(f'../Pickles/{split_type}_x_train.p', 'rb'))
    x_test = pickle.load(open(f'../Pickles/{split_type}_x_test.p', 'rb'))
    y_train = pickle.load(open(f'../Pickles/{split_type}_y_train.p', 'rb'))
    y_test = pickle.load(open(f'../Pickles/{split_type}_y_test.p', 'rb'))
    
    
    print('Train Value Counts')
    print(y_train.value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Test Value Counts')
    print(y_test.value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return x_train, x_test, y_train, y_test


def import_tweet_data():
    """Imports the tweet data from the 'data' folder, 
    with ISO-8859-1 encoding.
    
    Output: A Pandas DataFrame"""
    
    df = pd.read_csv('data/TweetsOriginal.csv', encoding = 'ISO-8859-1' )
    return df

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
    


def clean_split(split_type, df): 
    new_df = pd.DataFrame() 
    new_df['Text'] = df['tweet_text']
    new_df['Item'] = df['emotion_in_tweet_is_directed_at']
    new_df['Emotion'] = df['is_there_an_emotion_directed_at_a_brand_or_product']
    if split_type == 2: 
        new_df['Emotion_New'] = new_df.Emotion.map(encode_emotion_2)
    else: 
        new_df['Emotion_New'] = new_df.Emotion.map(encode_emotion_3)
    
    #dropping na in columns Text and Emotion
    new_df.dropna(subset = ['Text', 'Emotion_New'], inplace = True)
 
    tweet_token = TweetTokenizer()

    eng_words = set(words.words())
 
    tweets = new_df.Text.values
    new_tweets = []
    for sent in tweets:
        new_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",sent.lower()).split()))
    new_df.Text = new_tweets
    
    word_tokenizer = RegexpTokenizer("([a-zA-Z&]+(?:'[a-z]+)?)")
    word_lemmet = WordNetLemmatizer()
    word_stem = PorterStemmer()
    tweet_token = TweetTokenizer()
#     new_df.Text = new_df.Text.map(lambda x: tweet_token.tokenize(x.lower()))
#     new_df.Text = new_df.Text.map(lambda x: ' '.join(x))
    new_df.Text= new_df.Text.map(lambda x: word_tokenizer.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join([word_stem.stem(i) for i in x if len(i) > 2]))
#     new_df.Text = new_df.Text.map(lambda x: ' '.join([word_lemmet.lemmatize(i) for i in x if len(i) > 2]))

    
    
    
    if split_type == 2:
        print('Original Value Counts')
        print(new_df.Emotion_New.value_counts())
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pos_df = new_df[new_df.Emotion_New == 1]
        neg_df = new_df[new_df.Emotion_New == 0]
        
        resample_pos = resample(pos_df, n_samples = 600, random_state = 10, replace = False)
        new_df = resample_pos.append(neg_df, ignore_index = True)
        print('Final Resampled Value Counts')
        print(new_df.Emotion_New.value_counts())
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    else: 
        print('Original Value Counts')
        print(new_df.Emotion_New.value_counts())
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
           

        pos_df = new_df[new_df.Emotion_New == 1]
        neg_df = new_df[new_df.Emotion_New == 0]
        neut_df = new_df[new_df.Emotion_New == 2]

        resample_pos = resample(pos_df, n_samples = 600, random_state = 10, replace = False)
        resample_neut = resample(neut_df, n_samples = 600, random_state = 10, replace = False)
        
        new_df = neg_df.append(resample_pos, ignore_index = True)
        new_df = new_df.append(resample_neut, ignore_index = True)
        print('Final Resampled Value Counts')
        print(new_df.Emotion_New.value_counts())
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #split into test and trains
    x_train, x_test, y_train, y_test = train_test_split(new_df.Text, new_df.Emotion_New, stratify = new_df.Emotion_New,                                        
                                                        train_size = .85, random_state = 10)
    
    #removing stop words
    new_stop = ['abacus', 'yr', 'acerbic', 'bcet', 'beechwood', 'bicycle', 'brian', 'sxsw', 'ce', 'louis', 'mngr', 
               'rewardswagon', 'loui', 'csuitecsourc', 'wjchat', 'peter', 'bbq', 'au', 'austin', 'awesometim', 'bankinnov', 
                'barton', 'boooo', 'bookbook']
    stop = stopwords.words('english') + new_stop
    vectorizer= CountVectorizer(stop_words = stop, max_features = 6000, ngram_range=(1,2))
#     vectorizer= TfidfVectorizer(stop_words = stop, max_features = 5000, ngram_range=(1,2))
   
    
    clean_train = x_train.values
    clean_test = x_test.values
    vectorizer.fit(clean_train)
    pickle.dump(vectorizer, open('../Pickles/vectorizer.p', 'wb'))

    train_features =vectorizer.transform(clean_train).toarray()
    test_features = vectorizer.transform(clean_test).toarray()
    
    train_df = pd.DataFrame(train_features, columns = vectorizer.get_feature_names())
    train_df['target'] = y_train.values
    
    train_df.to_csv('data/TrainDF.csv', index = False)
    test_df = pd.DataFrame(test_features, columns = vectorizer.get_feature_names())
    test_df['target'] = y_test.values
    
    test_df.to_csv('data/TestDF.csv', index = False)
    #pickling
    pickle.dump(train_features, open(f'../Pickles/{split_type}_x_train.p', 'wb'))
    pickle.dump(test_features, open(f'../Pickles/{split_type}_x_test.p', 'wb'))
    pickle.dump(y_train, open(f'../Pickles/{split_type}_y_train.p', 'wb'))
    pickle.dump(y_test, open(f'../Pickles/{split_type}_y_test.p', 'wb'))
    
    print('Finished Pickling')
    
    return train_features, test_features, y_train, y_test, train_df
