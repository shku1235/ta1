import re
from collections import defaultdict
import operator

sentence = []
old_sentiment= []
old_text = []
text = []
unique_sentence_list = []
sentiment = []

data=open('train.tsv','r')
text_loaded=data.read()
split_text_rows=re.split(r'\n+',text_loaded)
for i in range(0,len(split_text_rows)):

    split_text=re.split(r'\t+', split_text_rows[i])
    for j in range(0, len(split_text)):
        if (j% 4 ==1):
            sentence.append(split_text[j])
        if (j % 4 == 2):
            old_text.append(split_text[j])
        if (j % 4 == 3):
            old_sentiment.append(split_text[j])



#print(len(old_text))
data.close()

old_text=old_text[1:]
old_sentiment=old_sentiment[1:]
sentence = sentence[1:]
#print(len(sentence))

for i in range(0,len(sentence)):
    if sentence[i] not in unique_sentence_list:
        text.append(old_text[i])
        sentiment.append(old_sentiment[i])
        unique_sentence_list.append(sentence[i])

print(len(text))

from nltk.corpus import stopwords
s=set(stopwords.words('english'))

for i in range (0,len(text)):
    text[i] = text[i].replace('.','')
    text[i] = text[i].replace(',', '')
#    text[i] = re.sub(r'\d+', 'number', text[i])
    text[i]= text[i].split(" ")
    filtered_words = list(filter(lambda word: word not in stopwords.words('english'), text[i]))
    text[i] = ' '.join(filtered_words)



new_sentiment_list= []
new_text_list = []

word_count = []
print("starting appending the lists")
for i in range(0,len(sentiment)):
    new_sentiment_list.append(sentiment[i])
    new_text_list.append(text[i])
    word_count.append(len(new_text_list))
from nltk import *
stemmer = SnowballStemmer("english")
for i in range (0,len(new_text_list)):
    new_text_list[i]=stemmer.stem(new_text_list[i])
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,4),   min_df= 0.001 , max_df= 0.4)

x = vectorizer.fit_transform(new_text_list)
import numpy as np
import pandas as pd
final_list_of_feature_names =  vectorizer.get_feature_names()
final_list_of_feature_values = x.toarray()

values_with_inserted_column_name = pd.DataFrame(final_list_of_feature_values,columns=final_list_of_feature_names)
values_with_inserted_column_name['Word_count'] = word_count
values_with_inserted_column_name['Target_class'] = new_sentiment_list

print(values_with_inserted_column_name.head())
values_with_inserted_column_name.to_csv('Output_train.csv')






