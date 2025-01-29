import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#sample emails
emails = [
    "Free money now",
    "Win a free lottery",
    "Hello friend, how are you?", 
    "Meeting at noon",
    "Win money now"
    ]

labels = [1, 1, 0, 0, 1]  #1 - spam, 0 - not spam

#convert the text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

print(X.toarray())

