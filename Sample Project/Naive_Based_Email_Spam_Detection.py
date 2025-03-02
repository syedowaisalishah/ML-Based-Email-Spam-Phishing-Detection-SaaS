#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[16]:


email = pd.read_csv("spam.csv")
print(email)
email.head()


# In[17]:


email.groupby('Category').describe()


# In[19]:


email['spam']=email['Category'].apply(lambda x: 1 if x=='spam' else 0)
email.head()


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(email.Message,email.spam,test_size=0.25)


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]


# In[24]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


# In[34]:


emails = [
    'Welcome to the new RISC-V International newsletter, where you can discover the latest from the RISC–V community, including news, events, technical developments, and training opportunities!',
    'Hot, mature women are waiting. Dont let them down. Join now with a free 7-day VIP pass! click the link'
]
emails_count = v.transform(emails)
model.predict(emails_count)


# In[35]:


X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)


# In[36]:


from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])


# In[37]:


clf.fit(X_train, y_train)


# In[38]:


clf.score(X_test,y_test)


# In[39]:


clf.predict(emails)


# In[40]:


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Remove stopwords & stem
    return text

email['cleaned_message'] = email['Message'].apply(clean_text)
email.head()


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # Convert top 5000 words
X = vectorizer.fit_transform(email['cleaned_message'])  # Transform cleaned text


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, email['spam'], test_size=0.25, random_state=42)

# Naïve Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"Naïve Bayes Accuracy: {nb_acc}")
print(f"Random Forest Accuracy: {rf_acc}")


# In[43]:


import pickle

# Save Naïve Bayes Model (or change to rf_model if preferred)
with open("spam_classifier.pkl", "wb") as model_file:
    pickle.dump(nb_model, model_file)

# Save TF-IDF Vectorizer (you need to find the variable name in your notebook, let's assume it's "vectorizer")
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


# In[ ]:




