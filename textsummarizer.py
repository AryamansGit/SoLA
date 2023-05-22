import spacy
from nltk.corpus import stopwords

# Reading Time
def readingTime(input):
    total_words = [token.text for token in nlp(input)]
    estimatedtime = len(total_words)/200  # 200 words per minute avg reading time
    return '{} mins'.format(estimatedtime)


stop_words = stopwords.words('english')
user_input = input("Enter text to summarise = ")
nlp = spacy.load("Dataset/en_core_web_lg/en_core_web_lg-3.5.0")
doc = nlp(user_input)

# create a word frequency list
word_freq_list = {}
for word in doc:
    if word.text not in stop_words :
        if word.text not in word_freq_list.keys() :
            word_freq_list[word.text] = 1
        else:
            word_freq_list[word.text] += 1

print("Word frequency = ", word_freq_list)

# convert to weighted frequencies to remove bias
max_freq = max(word_freq_list.values())
for word in word_freq_list.keys() :
    word_freq_list[word] = word_freq_list[word]/max_freq

print("Max freq = ", max_freq)
print("Normalized frequencies = ", word_freq_list)

# get sentence scores
sentence_list = [sentence for sentence in doc.sents]
print("Sentences = ", sentence_list)
sentence_scores = {}
for sentence in sentence_list:
    if len(sentence.text.split(" ")) < 25:  # large unreadable sentence
        for word in sentence :
            if word.text.lower() in word_freq_list.keys() :
                if sentence not in sentence_scores.keys() :
                    sentence_scores[sentence] = word_freq_list[word.text.lower()]
                else:
                    sentence_scores[sentence] += word_freq_list[word.text.lower()]

print("Sentence scores = ", sentence_scores)
avg_sentence_score = sum(sentence_scores.values())/len(sentence_scores)
print("Average sentence score = ", avg_sentence_score)

top = {}
for sentence in sentence_scores :
    if sentence_scores[sentence] >= avg_sentence_score :
        top[sentence] = sentence_scores[sentence]

summarisation = " ".join([sentence.text for sentence in top])
print("Result = ", summarisation)
print("Original Length = ", len(user_input))
print("Reading time of original = ", readingTime(user_input))
print("------------------------------------------------------------------------------------------------------------------")
print("Length after summarisation = ", len(summarisation))
print("Reading time of summarized text = ", readingTime(summarisation))