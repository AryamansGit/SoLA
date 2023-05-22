import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# set the seed so that sampling and model results are always the same
np.random.seed(80201)

# set dataset print options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)

#read the cleaned dataset (preprocessin done in preprcoessing.py)
clean_data = pd.read_csv("Dataset/cleandata.csv", delimiter=",", dtype={"Review" : str, "Sentiment" : int})

reviews = pd.DataFrame(clean_data)
print("All Data dimensions",reviews.shape)

# Dataset dimensions------------------------------------------------------------------------------------------------------------
print("Dataset Dimensions = ", reviews.shape)

# Dataset information-----------------------------------------------------------------------------------------------------------
print("Dataset statistics = ")
print(reviews.describe(include="all"))
print(reviews.info(verbose=True))


# Distribution of target variables---------------------------------------------------------------------------------------------------
print("Distribution of Target Variable = ")
print(reviews['Sentiment'].value_counts())

sns.histplot(x="Sentiment",shrink=0.5, data=reviews, binrange=(-1.5,1.5), binwidth=1)
plt.xticks([-1,1],["Negative","Positive"])
plt.show()


# most common words in the dataset--------------------------------------------------------------------------------------------------

#split into positive reviews and negative reviews
pos_reviews = reviews[reviews.Sentiment == 1]
print("Positive dataset dimensions = ", pos_reviews.shape)
neg_reviews = reviews[reviews.Sentiment == -1]
print("Negative dataset dimensions = ",neg_reviews.shape)

vec = CountVectorizer(binary=True)
dtm = vec.fit_transform(reviews["Review"])
count_array = dtm.toarray()
dtm_df =pd.DataFrame(data=count_array,columns=vec.get_feature_names_out())
sums = dtm_df.sum()
# print(sums.sort_values(ascending=False))
all_freq_dt= pd.DataFrame()
all_freq_dt["Words"] = vec.get_feature_names_out()
all_freq_dt["Count"] = sums.values
ordered_all_freq_dt= all_freq_dt.sort_values("Count",ascending=False)

#top 20 most frequent words in the dataset
print("Top 20 words in the dataset = ")
print(ordered_all_freq_dt.head(20))

#Wordlcoud of 50 most frequent words
data = dict(zip(ordered_all_freq_dt["Words"].tolist(),ordered_all_freq_dt["Count"].tolist()))
wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(data)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Graph of 20 most frequent words
sns.barplot(x="Words",y ="Count",data= ordered_all_freq_dt.head(20))
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Top 20 words in the dataset")
plt.show()

# top 20 words in positive reviews---------------------------------------------------------------------------------------------

vec_pos = CountVectorizer(binary=True)
dtm_pos = vec_pos.fit_transform(pos_reviews["Review"])
count_array_pos = dtm_pos.toarray()
dtm_df_pos =pd.DataFrame(data=count_array_pos,columns=vec_pos.get_feature_names_out())
sums_pos = dtm_df_pos.sum()
print(sums_pos.sort_values(ascending=False))
pos_freq_dt = pd.DataFrame()
pos_freq_dt["Words"] = vec_pos.get_feature_names_out()
pos_freq_dt["Count"] = sums_pos.values
ordered_pos_freq_dt = pos_freq_dt.sort_values("Count",ascending=False)

#top 20 most frequent words in the dataset
print("Top 20 words in positive reviews = ")
print(ordered_pos_freq_dt.head(20))

#Wordlcoud of 50 most frequent words
data_pos = dict(zip(ordered_pos_freq_dt["Words"].tolist(),ordered_pos_freq_dt["Count"].tolist()))
wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(data_pos)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Graph of 20 most frequent words
sns.barplot(x="Words",y ="Count",data= ordered_pos_freq_dt.head(20))
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Top 20 words in positive reviews")
plt.show()

# top 20 words in negative reviews----------------------------------------------------------------------------------------------

vec_neg = CountVectorizer(binary=True)
dtm_neg = vec_neg.fit_transform(neg_reviews["Review"])
count_array_neg = dtm_neg.toarray()
dtm_df_neg =pd.DataFrame(data=count_array_neg,columns=vec_neg.get_feature_names_out())
sums_neg = dtm_df_neg.sum()
print(sums_neg.sort_values(ascending=False))
neg_freq_dt = pd.DataFrame()
neg_freq_dt["Words"] = vec_neg.get_feature_names_out()
neg_freq_dt["Count"] = sums_neg.values
ordered_neg_freq_dt = neg_freq_dt.sort_values("Count", ascending=False)

#top 20 most frequent words in the dataset
print("Top 20 words in negative reviews = ")
print(ordered_neg_freq_dt.head(20))

#Wordlcoud of 50 most frequent words
data_neg = dict(zip(ordered_neg_freq_dt["Words"].tolist(),ordered_neg_freq_dt["Count"].tolist()))
wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(data_neg)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Graph of 20 most frequent words
sns.barplot(x="Words",y ="Count",data= ordered_neg_freq_dt.head(20))
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Top 20 words in negative reviews")
plt.show()