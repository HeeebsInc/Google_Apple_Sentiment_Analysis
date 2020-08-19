import pandas as pd
import numpy as np 
from tqdm import tqdm 
from pprint import pprint
import enchant
import pickle
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix


import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

def get_pickles(split_type = 0): 
    
    """
    Opens up the training and test data files. 
    
    Input:
    
    split_type: int. the split type of the file, based on if there are
    2 or 3 classes that are being used for prediction. 
    """
    
    if split_type > 0:
        split_type = str(split_type) + '_'
    else:
        split_type = ''
        
    x_train = pickle.load(open(f'../Pickles/{split_type}x_train.p', 'rb'))
    x_test = pickle.load(open(f'../Pickles/{split_type}x_test.p', 'rb'))
    y_train = pickle.load(open(f'../Pickles/{split_type}y_train.p', 'rb'))
    y_test = pickle.load(open(f'../Pickles/{split_type}y_test.p', 'rb'))
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
    

def clean_split(split_type, df):
    new_df = pd.DataFrame()
    
    new_df.rename(columns = {
        'Text' : 'tweet_text';
        'Item' : 'emotion_in_tweet_is_directed_at';
        'Emotion' : 'is_there_an_emotion_directed_at_a_brand_or_product'
    })
    
    new_df['Emotion_New'] = new_df.Emotion.replace(to_replace = {
        'negative emotion' : 0;
        'positive emotion' : 1;
        'No emotion toward brand or product': None
    }
    )
    
    #dropping na in columns Text and Emotion
    new_df.dropna(subset = ['Text', 'Emotion_New'], inplace = True)
    eng_words = set(nltk.corpus.words.words())
    tweets = new_df.Text.values
    new_tweets = []
    
    for sent in tweets:
        new_tweets.append(" ".join(w for w in nltk.wordpunct_tokenize(sent) \
                 if w.lower() in eng_words or not w.isalpha()))
        
    new_df.Text = new_tweets
    
    word_tokenizer = RegexpTokenizer("([a-zA-Z&]+(?:'[a-z]+)?)")
    word_lemmet = WordNetLemmatizer()
    tweet_token = TweetTokenizer()
    
    new_df.Text = new_df.Text.map(lambda x: tweet_token.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join(x))
    new_df.Text= new_df.Text.map(lambda x: word_tokenizer.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join([word_lemmet.lemmatize(i) for i in x]))
    
    print(f'''Original Value Counts
    {new_df.Emotion_New.value_counts()}
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ''')

    pos_df = new_df[new_df.Emotion_New == 1]
    neg_df = new_df[new_df.Emotion_New == 0]
    resample_pos = resample(pos_df, n_samples = 600, random_state = 10, replace = False)
    new_df = resample_pos.append(neg_df, ignore_index = True)
    
    print(f'''Final Resampled Value Counts
    {new_df.Emotion_New.value_counts()}
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ''')

    #split into test and trains
    x_train, x_test, y_train, y_test = train_test_split(new_df.Text, new_df.Emotion_New, stratify = new_df.Emotion_New,
                                                        train_size = .85, random_state = 10)
    #removing stop words
    stop = stopwords.words('english')
    vectorizer= CountVectorizer(stop_words = stop, max_features = 5000, ngram_range=(1,2))
    
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
    
    return train_features, test_features, y_train, y_test

 
    
def import_tweet_data():
    """Imports the tweet data from the 'data' folder, 
    with ISO-8859-1 encoding.
    
    Output: A Pandas DataFrame"""
    
    df = pd.read_csv('data/TweetsOriginal.csv', encoding = 'ISO-8859-1' )
    return df