import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import nltk 
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.utils import resample
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
import re


def plot_model_results(results, model_names, filepath, figsize = (10, 8)):
    
    """Plots and saves an image of the plot.
    Input:
    results: the results of the models, gained from test_models()
    model_names: the names of models used, gained from test_models()
    filepath: the filepath for the graph image to be saved to
    """
    
    plt.figure(figsize = figsize)
    plt.boxplot(results, labels = model_names, showmeans = True)
    plt.title('Accuracy for Each Vanilla Model')
    plt.ylabel('Accuracy'); plt.xlabel('Model')
    plt.savefig(filepath)
    plt.show()

def test_models(x_train, y_train, models, n_jobs = 2):
    """
    Test all models given.
    
    Hey sam I would appriciate it if you could add to this doc string!
    
    returns: results, model_names"""
    results = []
    model_names = []
    pbar = tqdm(models.items())
    
    for model, m in pbar: 
        pbar.set_description(f'Evaluating {model.upper()}')
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5)
        scores = cross_val_score(m, x_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = n_jobs, 
                                 error_score = 'raise')
        results.append(scores)
        model_names.append(model)
        
    return results, model_names


def stacked_model(models):
    """Creates a stacked model given a dictionary of SciKitLearn models
    -----------------------------------------
    Input: 
        models: Dictionary containing the model name and function.
    
    Output: 
        stack_model: A SciKitLearn StackingClassifier object
    -----------------------------------------"""

    stack_m = [] 
    for model, m in models.items(): 
        stack_m.append((model, m))
    stack_model = StackingClassifier(estimators = stack_m, final_estimator = LogisticRegression(), cv = 5)
    models['stacked'] = stack_model
    
    return stack_model


def save_cv_results(model_names, results, filename):
    """
    Pickles the model's results
    
    Input: 
    
    model_names: list of model names 
    results: list of results 
    filename: str, path for the file to be saved
    
    Output: 
    
    Pickle file, saved in the location specified in filename
    """
    vanilla_dict = {i:y for i,y in zip(model_names, results)}
    return pickle.dump(vanilla_dict, open(filename, 'wb'))


def import_tweet_data():
    """Imports the tweet data from the 'data' folder, 
    with ISO-8859-1 encoding.
    
    Output: A Pandas DataFrame"""
    
    df = pd.read_csv('data/TweetsOriginal.csv', encoding = 'ISO-8859-1' )
    return df
    


def clean_split(df): 
    '''This function takes in the original tweet dataframe and does the following
    
    1. Change column names
    2. Removes NA values within the columns Tweet and Emotion 
    3. Removes unwanted characters ($,@,#, etc.) 
    4. Applies a stemming technique to normalize the words
    5. Applies a count vectorizer on the train data 
    6. Save the count vectorizer as a pickle so that it can be used to transform new data 
    7. Save the train and test df to csv for visualizations
    8. Resample the train and test so the targets are balanced
    9. return x_train, x_test, y_train, y_test'''
    
    new_df = df
    
    new_df.rename(columns = {
        'tweet_text': 'Text', 
         'emotion_in_tweet_is_directed_at' : 'Item', 
        'is_there_an_emotion_directed_at_a_brand_or_product' : 'Emotion'
    }, inplace = True)
    
    new_df['Emotion_New'] = new_df.Emotion.replace(to_replace = {
        'Negative emotion' : 0,
        'Positive emotion' : 1,
        'No emotion toward brand or product': None,
        "I can't tell" : None
    }
    )
    
    #dropping na in columns Text and Emotion
    new_df.dropna(subset = ['Text', 'Emotion_New'], inplace = True)
 
    tweet_token = TweetTokenizer()

    eng_words = set(words.words())
 
    tweets = new_df.Text.values
    new_tweets = []
    for sent in tweets:
        new_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",sent.lower()).split()))
    new_df.Text = new_tweets
    
    #removes unwanted characters
    word_tokenizer = RegexpTokenizer("([a-zA-Z&]+(?:'[a-z]+)?)")
    word_stem = PorterStemmer()
    tweet_token = TweetTokenizer()
    new_df.Text= new_df.Text.map(lambda x: word_tokenizer.tokenize(x.lower()))
    #includes only stemmed words
    new_df.Text = new_df.Text.map(lambda x: ' '.join([word_stem.stem(i) for i in x if len(i) > 2]))

    
    print('Original Value Counts')
    print(new_df.Emotion_New.value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    pos_df = new_df[new_df.Emotion_New == 1]
    neg_df = new_df[new_df.Emotion_New == 0]

    #resample data based on target
    resample_pos = resample(pos_df, n_samples = 600, random_state = 10, replace = False)
    new_df = resample_pos.append(neg_df, ignore_index = True)
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
   
    #creates a test and train df for visualization
    clean_train = x_train.values
    clean_test = x_test.values
    vectorizer.fit(clean_train)
    
    #saves vectorizer as a pickle for future use
    pickle.dump(vectorizer, open('../Pickles/vectorizer.p', 'wb'))

    train_features =vectorizer.transform(clean_train).toarray()
    test_features = vectorizer.transform(clean_test).toarray()
    
    train_df = pd.DataFrame(train_features, columns = vectorizer.get_feature_names())
    train_df['target'] = y_train.values
    
    test_df = pd.DataFrame(test_features, columns = vectorizer.get_feature_names())
    test_df['target'] = y_test.values
    
    #saves test and train df for visualizations
    train_df.to_csv('data/TrainDF.csv', index = False)
    test_df.to_csv('data/TestDF.csv', index = False)
   
    #return x_train, x_test, y_train, y_test
    return train_features, test_features, y_train, y_test