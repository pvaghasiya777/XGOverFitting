import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Email_Spam.csv')

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    max_depth=5,          # Limit the depth of trees
    min_child_weight=1,   # Minimum sum of instance weight in a child
    subsample=0.8,        # Fraction of samples used for growing trees
    reg_alpha=0.1,        # L1 regularization
    reg_lambda=0.1        # L2 regularization
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(report)