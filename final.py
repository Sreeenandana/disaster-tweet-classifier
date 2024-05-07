import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Data is loaded
data = pd.read_csv(r"C:\Users\kkbmu\OneDrive\Desktop\mini project\Code\train2.csv")

'''Here URLs,non-alphanumeric characters and whitespaces are removed. The cleaned text is then tokenised. 
   Stop words are removed from those tokens and then lemmatised. lemmatised tokens are then joined and returned.'''

stop_words = set(stopwords.words('english'))
stop_words.update(["like", "u", "û_", "amp"])
lemmatizer = WordNetLemmatizer()

# preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+|\b\d+\b|\W", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text= text.strip() 
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    filtered_tokens = [word for word in tokens if word not in stop_words] 
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens) 
data['text'] = data['text'].apply(preprocess_text)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.1, random_state=43)

'''Feature extraction and vectorisation '''
tfidf_vectorizer = TfidfVectorizer(min_df=7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


'''logistic '''
print("LOGISTIC REGRESSION")
'''training the model'''
logistic_clf_tfidf = LogisticRegression(solver="liblinear")
#logistic_clf_tfidf = LogisticRegression(solver="liblinear")
logistic_clf_tfidf.fit(X_train_tfidf, y_train)

'''evaluation'''

#training accuracy
y_train_log = logistic_clf_tfidf.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_log)
print("Training Accuracy:{:.2f}".format(train_accuracy))

#testing accuracy
y_test_log = logistic_clf_tfidf.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_log)

print("Testing Accuracy : {:.2f}".format(test_accuracy))

# classification report for testing data
print("Classification Report for Testing Data:\n", classification_report(y_test, y_test_log))


# Save the model to a file using pickle
with open('finalmodel.pkl', 'wb') as f:
    pickle.dump(logistic_clf_tfidf, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)


'''SVM'''
print("SVM")
from sklearn.svm import SVC

# Create an SVC model with class weights
svc_clf_tfidf = SVC(class_weight={0: 0.4, 1: 0.6})

# Train the model
svc_clf_tfidf.fit(X_train_tfidf, y_train)

'''Evaluation'''
# Training accuracy
y_train_svm = svc_clf_tfidf.predict(X_train_tfidf)
train_accuracy_svm = accuracy_score(y_train, y_train_svm)
print("Training Accuracy (SVM): {:.2f}".format(train_accuracy_svm))

# Testing accuracy
y_test_svm = svc_clf_tfidf.predict(X_test_tfidf)
test_accuracy_svm = accuracy_score(y_test, y_test_svm)
print("Testing Accuracy (SVM): {:.2f}".format(test_accuracy_svm))

# Print classification report for testing data
print("Classification Report for Testing Data (SVM):\n", classification_report(y_test, y_test_svm))

'''Random Forest'''

print("RANDOM FOREST")
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model with tuned parameters
rf_clf_tfidf = RandomForestClassifier(min_samples_split=29, min_samples_leaf=3)

# Train the model
rf_clf_tfidf.fit(X_train_tfidf, y_train)

'''Evaluation'''
# Training accuracy
y_train_rf = rf_clf_tfidf.predict(X_train_tfidf)
train_accuracy_rf = accuracy_score(y_train, y_train_rf)
print("Training Accuracy (Random Forest): {:.2f}".format(train_accuracy_rf))

# Testing accuracy
y_test_rf = rf_clf_tfidf.predict(X_test_tfidf)
test_accuracy_rf = accuracy_score(y_test, y_test_rf)
print("Testing Accuracy (Random Forest): {:.2f}".format(test_accuracy_rf))

# Print classification report for testing data
print("Classification Report for Testing Data (Random Forest):\n", classification_report(y_test, y_test_rf))


'''ensemble'''

print("ENSEMBLE")
from sklearn.ensemble import VotingClassifier

# Define individual classifiers
logistic_clf_tfidf = LogisticRegression(solver="liblinear")
svc_clf_tfidf = SVC(class_weight={0: 0.4, 1: 0.6})

# Create ensemble
ensemble_clf = VotingClassifier(estimators=[('logistic', logistic_clf_tfidf), ('svm', svc_clf_tfidf)], voting='hard')

# Train ensemble
ensemble_clf.fit(X_train_tfidf, y_train)

'''evaluation'''
# Training accuracy
train_accuracy = ensemble_clf.score(X_train_tfidf, y_train)
print("Training Accuracy: {:.2f}".format(train_accuracy))

# Testing accuracy
test_accuracy = ensemble_clf.score(X_test_tfidf, y_test)
print("Testing Accuracy: {:.2f}".format(test_accuracy))

# Print classification report for testing data
y_test_pred = ensemble_clf.predict(X_test_tfidf)
print("Classification Report for Testing Data:\n", classification_report(y_test, y_test_pred))

