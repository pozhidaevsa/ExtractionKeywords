import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

with open('text.txt', 'r') as file:
    data = file.read().replace('\n', '')

def pre_process(text):
    text=text.lower()
    text=re.sub("</?.*?>"," <> ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    text = [word for word in text.split() if word not in stopwords.words('russian')]
    stemmer = SnowballStemmer("russian") 
    lem = [stemmer.stem(word) for word in text]
    text = " ".join(lem)
    return text

data = pre_process(data)
docs = data.split('delimiter')

cv = CountVectorizer()
word_count_vector = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

feature_names=cv.get_feature_names()
tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[0]]))
sorted_items=sort_coo(tf_idf_vector.tocoo())
keywords=extract_topn_from_vector(feature_names,sorted_items, 20)

for k in keywords:
    print(k,keywords[k])

import csv
with open('output.csv', 'w', encoding='utf-16') as output:
    writer = csv.writer(output)
    for key, value in keywords.items():
        writer.writerow([key, value])