import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as p
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve,roc_auc_score
import joblib
from sklearn.model_selection import train_test_split


def plotROC(y_test, y_pred):

    # set the seed so that sampling and model results are always the same
    np.random.seed(80201)

    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    auc = roc_auc_score(y_test,y_pred)
    roc = plt.figure()
    plt.plot(fpr,tpr,label="AUC = %0.3f" % auc)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    return auc, roc

def allPlots():
    # set the seed so that sampling and model results are always the same
    np.random.seed(80201)

    #read the cleaned dataset (preprocessin done in preprcoessing.py)
    clean_data = pd.read_csv("Dataset/cleandata.csv", delimiter=",", dtype={"Review" : str, "Sentiment" : int})
    labels = clean_data["Sentiment"]
    # Load the model from the saved pickle file
    SVMClassifier = joblib.load('SVMClassifier2.pkl')
    NBClassifier = joblib.load('NBClassifier.pkl')
    RFClassifier = joblib.load('RFCClassifier.pkl')

    vectorizer = TfidfVectorizer(min_df=15)
    vec = vectorizer.fit_transform(clean_data["Review"])

    X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2, stratify= labels)

    y_pred_RF = RFClassifier.predict(X_test)
    y_pred_NB = NBClassifier.predict(X_test)
    y_pred_SVM = SVMClassifier.predict(X_test)

    fpr,tpr,threshold = roc_curve(y_test,y_pred_RF)
    auc_RF = roc_auc_score(y_test,y_pred_RF)
    roc = plt.figure()
    plt.plot(fpr,tpr,label="Random Forest, AUC = %0.3f" % auc_RF)

    fpr,tpr,threshold = roc_curve(y_test,y_pred_NB)
    auc_NB = roc_auc_score(y_test,y_pred_NB)
    plt.plot(fpr,tpr,label="Naive Bayes, AUC = %0.3f" % auc_NB)

    fpr,tpr,threshold = roc_curve(y_test,y_pred_SVM)
    auc_SVM = roc_auc_score(y_test,y_pred_SVM)
    plt.plot(fpr,tpr,label="Support Vector Machine, AUC = %0.3f" % auc_SVM)

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    roc.show()





