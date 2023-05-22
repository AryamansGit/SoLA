import pandas as pd
from time import time
import numpy as np
import spacy

from joblib import Parallel, delayed

# set the seed so that sampling and model results are always the same
np.random.seed(80201)

# set dataset print options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)

# read dataset
df = pd.read_csv("Dataset/train_reviews.csv", delimiter=",", names=["Stars", "Title", "Review"],
                 dtype={"Stars": int, "Title": str, "Review": str})

# create a pandas dataframe from input dataset
reviews = pd.DataFrame(df)

t=time()
# After much analysis, neutral(3) star reviews are dropped as they were skewing model results.
reviews = reviews.drop(reviews[reviews.Stars == 3].index)
print("Dataset Dimensions = ", reviews.shape)

# title column is not useful
del reviews['Title']

print("Dataset Preview = ")
print(reviews.head(5))
print(reviews['Stars'].value_counts())

# dataset is quite large so processing is done on a sample of the dataset to preserve computational resources
sampleofreviews = reviews.sample(frac=.025)
print(sampleofreviews.shape)
#sampleofreviews = reviews.head(10)

# convert the Stars column into -1 for negative reviews(1/2 star reviews) and 1 for positive reviews(/4/5 star reviews)
Sentiments = []
for ind in sampleofreviews.index:
    if sampleofreviews["Stars"][ind] < 3:
        Sentiments.append(-1)
    elif sampleofreviews["Stars"][ind] > 3:
        Sentiments.append(1)

sampleofreviews["Sentiment"] = Sentiments
print("Target Column Transformed")

# stars column is no longer required as it has been transformed into sentiment column
del sampleofreviews["Stars"]

# convert all review data to lowercase
sampleofreviews['Review'] = sampleofreviews['Review'].str.lower()
print("Reviews converted to lowercase")

# remove all punctuation from review data
sampleofreviews["Review"] = sampleofreviews["Review"].replace("[\'\"\\\/\,\.\;\:\!\?\@]", '', regex=True).astype(str)
print("Removed Punctuation")

# remove all numerical data from review data
sampleofreviews["Review"] = sampleofreviews["Review"].replace('\d+', '', regex=True).astype(str)
print("Removed all digits")


# Import stopwords with nltk.
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
sampleofreviews['Review'] = sampleofreviews['Review'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop]))
print("Stopwords removed")

# lemmatisation using spacy
nlp = spacy.load("Dataset/en_core_web_sm/en_core_web_sm-3.5.0")  # load an existing English template
sampleofreviews['Review'] = sampleofreviews['Review'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x)]))
print("Lemmatization done")

preprocessing_time = time() - t
print("Preprocessing time: %0.3fs" % preprocessing_time)

print("Dataset Preview = ")
print(sampleofreviews.head(5))

# save the cleaned dataset for later use
sampleofreviews.to_csv(path_or_buf="Dataset/cleandata.csv", index=False)

print("Dataset Dimensions = ", sampleofreviews.shape)
