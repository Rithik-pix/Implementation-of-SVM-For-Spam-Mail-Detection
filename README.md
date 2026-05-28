# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and label messages as spam or not spam.

2.Preprocess the text (remove punctuation, lowercase, and convert to numerical form using TF-IDF).

3.Split the data into training and testing sets.

4.Train an SVM classifier (e.g., LinearSVC) on the training data.

5.Predict and evaluate performance using accuracy and confusion matrix. 
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rithik Ram S
RegisterNumber: 212224230229
*/
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]           # keep only relevant columns
df.columns = ['label', 'message']

# Step 2: Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Text vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train SVM model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

## Output:
<img width="510" height="297" alt="image" src="https://github.com/user-attachments/assets/a0ef38cd-ed58-4040-9fec-cb1018b467c9" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
