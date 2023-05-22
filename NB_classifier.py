import pandas as pd
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB,NB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import numpy as np
import joblib
from createROC import plotROC

# set the seed so that sampling and model results are always the same
np.random.seed(80201)

#read the cleaned dataset (preprocessin done in preprcoessing.py)
clean_data = pd.read_csv("Dataset/cleandata.csv", delimiter=",", dtype={"Review" : str, "Sentiment" : int})

print("Dataset Preview = ")
print(clean_data.head(5))

# convert the dataframe into a document term matrix using the concept of TF-IDF/Count(TF) vectorization
NBvectorizer = TfidfVectorizer(min_df=15)
#NBvectorizer = CountVectorizer(min_df=9)
vec = NBvectorizer.fit_transform(clean_data["Review"])
labels = clean_data["Sentiment"]

print("Document Term Matrix Dimensions = ")
print(vec.shape)

# split the data into test and train subsets
X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2, stratify= labels)

t =time()

# Creating a Naive Bayes Classifier
NBclassifier = MultinomialNB()
NBclassifier.fit(X_train,y_train)

training_time = time() - t
print("Training time: %0.3fs" % training_time)

t = time()
y_pred = NBclassifier.predict(X_test)
testing_time = time() - t
print("Testing time: %0.3fs" % testing_time)

# NB Model Metrics
cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, -1])
disp.plot()
plt.show()

print("Model Accuracy = " + str(NBclassifier.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
kappa = cohen_kappa_score(y_test,y_pred)
print("Kappa = " , kappa)

auc,roc =plotROC(y_test,y_pred)
print("Area under the Curve = ", auc)
roc.show()

# Save the model as a pickle in a file

joblib.dump(NBclassifier, 'NBClassifier.pkl')
print("Model saved successfully")

# Save the fitted vectorizer as a pickle in a file

joblib.dump(NBvectorizer, 'FittedNBVectorizer.pkl')
print("Fitted Vectorizer saved successfully")