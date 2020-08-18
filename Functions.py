import pandas as pd
import numpy as np 
from tqdm import tqdm 
from pprint import pprint
import enchant
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample


import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords

def plot_model_results(results, model_names, filepath):
    
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

def test_models(models, n_jobs = 2):
    """
    Test all models given.
    
    Hey sam I would appriciate it if you could add to this doc string!"""
    results = []
    model_names = []
    pbar = tqdm(models.items())
    
    for model, m in pbar: 
        pbar.set_description(f'Evaluating {model.upper()}')
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5)
        scores = cross_val_score(m, x_train_new, y_train_new, scoring = 'accuracy', cv = cv, n_jobs = n_jobs, 
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

def resample(x_train, y_train, num_targets, n_samples):
    
    """Allows resampling of a dataframe as long as there are 2 or 3 targets. 
    Resamples all targets.
    
    Input: 
    
    x_train: array of features
    y_train: 1D array of targets
    num_targets: the number of targets, 2 or 3
    n_samples: how many samples will be chosen
    
    Output: 
    X_train: A pandas DataFrame
    y_train: A numpy array object
    """
    
    print(f"""Original Train Value Counts
{y_train.value_counts()}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Original Test Value Counts
{y_test.value_counts()}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~""")

    train_df = pd.DataFrame(x_train)
    train_df['target'] = y_train.values 
    
    neg_df = train_df[train_df.target == 0]
    pos_df = train_df[train_df.target == 1]

    resample_pos = resample(pos_df, n_samples = n_samples, random_state = 10, replace = False)
    resample_neg = resample(neg_df, n_samples = n_samples, random_state = 10, replace = False)
    
    df = resample_neg.append(resample_pos, ignore_index = True)
    
    if num_targets == 3:
        neut_df = train_df[train_df.target == 2]
        resample_neut = resample(neut_df, n_samples = n_samples, random_state = 10, replace = False)
        df = df.append(resample_neut, ignore_index = True)
    
    X_train = df.drop(columns = 'target')
    y_train = df.target.values.ravel()
    
    print(f"""Final Resampled Value Counts
{final_df.target.value_counts()}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~""")
    
    return X_train, y_train


def resample_skip_neg(x_train, y_train, num_targets, n_samples):
    
        """Allows resampling of a dataframe as long as there are 2 or 3 targets. 
    Resamples all targets except for those with a "0", the negative sentiment
    value. 
    
    Input: 
    
    x_train: array of features
    y_train: 1D array of targets
    num_targets: the number of targets, 2 or 3
    n_samples: how many samples will be chosen
    
    Output: 
    X_train: A pandas DataFrame
    y_train: A numpy array object
    """
    
    print(f"""Original Train Value Counts
{y_train.value_counts()}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Original Test Value Counts
{y_test.value_counts()}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~""")

    train_df = pd.DataFrame(x_train)
    train_df['target'] = y_train.values 
    
    neg_df = train_df[train_df.target == 0]
    pos_df = train_df[train_df.target == 1]

    resample_pos = resample(pos_df, n_samples = n_samples, random_state = 10, replace = False)
    
    df = neg_df.append(resample_neg, ignore_index = True)
    
    if num_targets == 3:
        neut_df = train_df[train_df.target == 2]
        resample_neut = resample(neut_df, n_samples = n_samples, random_state = 10, replace = False)
        df = df.append(resample_neut, ignore_index = True)
    
    X_train = df.drop(columns = 'target')
    y_train = df.target.values.ravel()
    
    print(f"""Final Resampled Value Counts
{final_df.target.value_counts()}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~""")
    
    return X_train, y_train

    
def get_pickles(split_type = 0): 
    
    """
    Opens up the training and test data files. 
    
    Input:
    
    split_type: int. the split type of the file, based on if there are
    2 or 3 classes that are being used for prediction. 
    """
    
    if split_type > 0:
        split_type = str(split_type) + '_'
        
    x_train = pickle.load(open(f'Pickles/{split_type}x_train.p', 'rb'))
    x_test = pickle.load(open(f'Pickles/{split_type}x_test.p', 'rb'))
    y_train = pickle.load(open(f'Pickles/{split_type}y_train.p', 'rb'))
    y_test = pickle.load(open(f'Pickles/{split_type}y_test.p', 'rb'))
    return x_train, x_test, y_train, y_test

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
    
    
    return train_features, test_features, y_train, y_test
 