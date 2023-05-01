import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize
# from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk.stem import PorterStemmer
from nltk import word_tokenize

def remove_whitespace(text):
    return  text.lstrip().rstrip()

def remove_stopwords(text):
    result = []
    en_stopwords = stopwords.words('english')
    for token in text:
        if token not in en_stopwords:
            result.append(token)          
    return result

def remove_punct(text): 
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

def lemmatization(text):
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()      
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'          
        result.append(wordnet.lemmatize(token,pos))  
    return result

def stemming(text):
    porter = PorterStemmer()   
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result

def preprocess_text(text):
  # for text in df[col].values:
    text = str(text)
    text = text[1:-1] if len(text)>0 and text[0]=='[' else text
    text = remove_whitespace(text)
    text=word_tokenize(text)
    text = remove_stopwords(text)
    text = remove_punct(text)
    text = lemmatization(text)
    # text = stemming(text)
    # print(text)
    return text