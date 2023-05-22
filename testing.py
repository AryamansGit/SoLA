import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

# set the seed so that sampling and model results are always the same
np.random.seed(80201)

#read the cleaned dataset (preprocessin done in preprcoessing.py)
clean_data = pd.read_csv("Dataset/cleandata.csv", delimiter=",", dtype={"Review" : str, "Sentiment" : int})
clean_data.dropna(how="any",inplace=True)
print(clean_data.shape)

# convert the dataframe into a document term matrix using the concept of TF-IDF vectorization
vectorizer = TfidfVectorizer(min_df=15)
vec = vectorizer.fit_transform(clean_data["Review"])
labels = clean_data["Sentiment"]

print("Document Term Matrix Dimensions = ")
print(vec.shape)


#perform pca
svd =TruncatedSVD(0.95)
X_svd = svd.fit_transform(vec)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_)
print(X_svd.shape)