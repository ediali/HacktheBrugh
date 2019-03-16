import pandas as pd
import re
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

####################################################################


def clean_text(df: object, text_field: object) -> object:
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|"
                                                              r""r"(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


def clean_sentence(sentence):
    phrase = sentence.lower()
    phrase = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|"r"(\w+:\/\/\S+)|^rt|http.+?", "", phrase)
    return phrase


train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

test_clean = clean_text(test, "tweet")
train_clean = clean_text(train, "tweet")

train_majority = train_clean[train_clean.label == 0]
train_minority = train_clean[train_clean.label == 1]

train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(train_majority), random_state=123)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
train_upsampled['label'].value_counts()

train_majority = train_clean[train_clean.label == 0]
train_minority = train_clean[train_clean.label == 1]

train_majority_downsampled = resample(train_majority, replace=True, n_samples=len(train_minority), random_state=123)
train_downsampled = pd.concat([train_majority_downsampled, train_minority])
train_downsampled['label'].value_counts()

train_all = pd.concat([train_upsampled, train_downsampled])
train_all['label'].value_counts()
############################################################

pipeline_sgd = Pipeline([ ('vect', CountVectorizer()), ('tfidf',  TfidfTransformer()), ('nb', SGDClassifier(loss='log')), ])

X_train, X_test, y_train, y_test = train_test_split(train_upsampled['tweet'], train_upsampled['label'], random_state=0)
model = pipeline_sgd.fit(train_upsampled['tweet'], train_upsampled['label'])
y_predict = model.predict(X_test)
print(f1_score(y_test, y_predict))

####################################################################


def check_hateful():
    comment = input("Enter a sentence: ")
    clean_comment = clean_sentence(comment)
    toPredict = []
    toPredict.append(clean_comment)
    y_predict = model.predict_proba(toPredict)
    if y_predict[0][1] > .60:
        print(y_predict[0][1])
        return "Hateful"


x = 0
count_hateful = 0
count_good = 0

while x < 5:
    if check_hateful() == "Hateful":
        count_hateful += 1
    else:
        count_good += 1
    x += 1

# Data to plot
labels = 'Hateful', 'Benign'
sizes = [count_hateful, count_good]
colors = ['gold', 'lightskyblue']
explode = (0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


####################################################################
