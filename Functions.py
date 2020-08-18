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
import pickle
from sklearn.utils import resample
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import itertools
import seaborn as sns



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
 

    eng_words = set(nltk.corpus.words.words())
    tweets = new_df.Text.values
    new_tweets = []
    for sent in tweets:
        new_tweets.append(" ".join(w for w in nltk.wordpunct_tokenize(sent) \
                 if w.lower() in eng_words or not w.isalpha()))
    new_df.Text = new_tweets
    
    
    
    word_tokenizer = RegexpTokenizer("([a-zA-Z&]+(?:'[a-z]+)?)")
    word_lemmet = WordNetLemmatizer()
    word_stemm = PorterStemmer()
    tweet_token = TweetTokenizer()
    new_df.Text = new_df.Text.map(lambda x: tweet_token.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join(x))
    new_df.Text= new_df.Text.map(lambda x: word_tokenizer.tokenize(x.lower()))
    new_df.Text = new_df.Text.map(lambda x: ' '.join([word_lemmet.lemmatize(i) for i in x]))
    
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
    stop = stopwords.words('english')
    vectorizer= CountVectorizer(stop_words = stop, max_features = 5000, ngram_range=(1,2))
#     vectorizer= TfidfVectorizer(stop_words = stop, max_features = 5000, ngram_range=(1,2))
   
    
    random_sent = 'I hate apple'
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
    
    return train_features, test_features, y_train, y_test




def plot_loss(model_history, model_type, act): 
    train_loss = model_history.history['loss']
    train_acc = model_history.history['acc']
    test_loss = model_history.history['val_loss']
    test_acc = model_history.history['val_acc']
    epochs = [i for i in range(1, len(test_acc)+1)]

    fig, ax = plt.subplots(1,2, figsize = (15,5))
    ax[0].plot(epochs, train_loss, label = 'Train Loss')
    ax[0].plot(epochs, test_loss, label = 'Test Loss')
    ax[0].set_title('Train/Test Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label = 'Train Accuracy')
    ax[1].plot(epochs, test_acc, label = 'Test Accuracy')
    ax[1].set_title('Train/Test Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    plt.savefig(f'figures/{model_type}_{act}_loss.png')

    

def get_roc_auc(model,model_type, act, x_train, y_train, x_test, y_test): 
    fig, ax = plt.subplots(1,2, figsize = (15,6))
    #AUC CURVE
    y_test_prob = model.predict(x_test)

    y_test_precision, y_test_recall, spec = precision_recall_curve(y_test, y_test_prob)
    y_test_predict = np.where(y_test_prob >= .5, 1, 0).ravel()
    y_test_f1= f1_score(y_test, y_test_predict)
    y_test_auc = auc(y_test_recall, y_test_precision)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    ax[0].plot(y_test_recall, y_test_precision, marker='.', label='CNN')
    ax[0].plot([0, 1], [no_skill, no_skill], linestyle='--', label='50/50', color = 'Black')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title(f'AUC Curve')
    ax[0].legend()

    #ROC CURVE
    ns_probs = [0 for i in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    y_test_roc = roc_auc_score(y_test, y_test_prob)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    y_test_fpr, y_test_tpr, threshold = roc_curve(y_test, y_test_prob)
    ax[1].plot(ns_fpr, ns_tpr, linestyle='--', label='50/50')
    ax[1].plot(y_test_fpr, y_test_tpr, marker='.', label='CNN')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'ROC Curve')
    ax[1].legend()
    plt.savefig(f'figures/{model_type}_{act}_ROCAUC.png')

    plt.show()
    

    print({'F1 Score': round(y_test_f1, 3), 'AUC': round(y_test_auc, 3), 'ROC':round(y_test_roc, 3)})
    
    
    

def plot_cm(y_test,y_train, y_train_prob, y_test_prob,thresholds, classes, model_type, act,
                          cmap=plt.cm.Blues):
    fig, ax = plt.subplots(len(thresholds),2, figsize = (10,10))

    for idx, thresh in enumerate(thresholds):
        y_test_predict = np.where(y_test_prob >= thresh, 1, 0)
        y_train_predict = np.where(y_train_prob >= thresh, 1, 0)
        train_cm = confusion_matrix(y_train, y_train_predict) 
        test_cm = confusion_matrix(y_test, y_test_predict)
        
        #test confusion
        ax[idx, 0].imshow(test_cm,  cmap=plt.cm.Blues) 

        ax[idx, 0].set_title(f'Test: Confusion Matrix | Threshold: {thresh}')
        ax[idx, 0].set_ylabel('True label')
        ax[idx, 0].set_xlabel('Predicted label')

        class_names = classes 
        tick_marks = np.arange(len(class_names))
        ax[idx, 0].set_xticks(tick_marks)
        ax[idx,0].set_xticklabels(class_names)
        ax[idx, 0].set_yticks(tick_marks)
        ax[idx, 0].set_yticklabels(class_names)

        th = test_cm.max() / 2. 

        for i, j in itertools.product(range(test_cm.shape[0]), range(test_cm.shape[1])):
                ax[idx, 0].text(j, i, f'{test_cm[i, j]}',# | {int(round(test_cm[i,j]/test_cm.ravel().sum(),5)*100)}%',
                         horizontalalignment='center',
                         color='white' if test_cm[i, j] > th else 'black')
        ax[idx, 0].set_ylim([-.5,1.5])
        
        #TRAIN CONFUSION
        ax[idx, 1].imshow(train_cm,  cmap=plt.cm.Blues) 

        ax[idx, 1].set_title(f'Train: Confusion Matrix | Threshold: {thresh}')
        ax[idx, 1].set_ylabel('True label')
        ax[idx, 1].set_xlabel('Predicted label')

        class_names = classes 
        tick_marks = np.arange(len(class_names))
        ax[idx, 1].set_xticks(tick_marks)
        ax[idx,1].set_xticklabels(class_names)
        ax[idx, 1].set_yticks(tick_marks)
        ax[idx, 1].set_yticklabels(class_names)


        th = train_cm.max() / 2. 

        for i, j in itertools.product(range(train_cm.shape[0]), range(train_cm.shape[1])):
                ax[idx, 1].text(j, i, f'{train_cm[i, j]}',# | {int(round(train_cm[i,j]/train_cm.ravel().sum(),5)*100)}%',
                         horizontalalignment='center',
                         color='white' if train_cm[i, j] > th else 'black')
        ax[idx, 1].set_ylim([-.5,1.5])
    plt.tight_layout()
    plt.savefig(f'figures/{model_type}_{act}_cf.png')
 
    plt.show()
