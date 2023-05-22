import pandas as pd
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm
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
clean_data.dropna(how="any",inplace=True)
#clean_data= clean_data.sample(frac=.1)

# convert the dataframe into a document term matrix using the concept of TF-IDF/Count(TF) vectorization
SVMvectorizer = TfidfVectorizer(min_df=15, ngram_range=(1,2))
# SVMvectorizer = TfidfVectorizer(min_df=15)

vec = SVMvectorizer.fit_transform(clean_data["Review"])
labels = clean_data["Sentiment"]

print("Document Term Matrix Dimensions = ")
print(vec.shape)

# svd =TruncatedSVD(n_components=10000)
# X_svd = svd.fit_transform(vec)
# variance_explained = svd.explained_variance_.sum()
# print(variance_explained)

# split the data into test and train subsets
#X_train, X_test, y_train, y_test = train_test_split(X_svd, labels, test_size=0.2, stratify= labels)
X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2, stratify= labels)

# parameter_grid = {
#     'C': [0.1, 1, 5, 10]
# }

t =time()
# n_estimators=10
# Creating a Support Vector Machine Classifier
OptimisedSVMClassifier = svm.SVC(kernel="linear", gamma="auto")
# SVMclassifier = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), n_jobs=1, max_samples=1.0/n_estimators, n_estimators=n_estimators))
# gs = GridSearchCV(SVMClassifier,parameter_grid, cv=3, return_train_score=False,n_jobs=6)
# print("Grid Search done")

OptimisedSVMClassifier.fit(X_train,y_train)
# gs.fit(X_train,y_train)
# OptimisedSVMClassifier = gs.best_estimator_
training_time = time() - t
# results = pd.DataFrame(gs.cv_results_)
# results.to_csv(path_or_buf="Dataset/optimisationResults.csv",index=False)
# print(results)
print("Training time: %0.3fs" % training_time)

t = time()
y_pred = OptimisedSVMClassifier.predict(X_test)
testing_time = time() - t
print("Testing time: %0.3fs" % testing_time)

# NB Model Metrics
cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, -1])
disp.plot()
plt.show()

print("Model Accuracy = " + str(OptimisedSVMClassifier.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
kappa = cohen_kappa_score(y_test,y_pred)
print("Kappa = " , kappa)

auc,roc =plotROC(y_test,y_pred)
print("Area under the Curve = ", auc)
roc.show()

# Save the model as a pickle in a file

joblib.dump(OptimisedSVMClassifier, 'SVMClassifier.pkl')
print("Model saved successfully")
#
# Save the fitted vectorizer as a pickle in a file

joblib.dump(SVMvectorizer, 'FittedSVMVectorizer.pkl')
print("Fitted Vectorizer saved successfully")