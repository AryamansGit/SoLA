import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import re


def loadmodel():
    #nltk.download('punkt')
    # Load the model from the saved pickle file
    classifier = joblib.load('SVMClassifier.pkl')

    # Load the fitted vectorizer from saved pickle file
    vectorizer = joblib.load('FittedSVMVectorizer.pkl')
    return classifier, vectorizer

def sentimentAnalyser(classifier, vectorizer, user_input):
    # Test model against user input data
    user_input = user_input.lower()
    user_input = re.sub("[^\w\s]","",user_input)

    # Import stopwords with nltk.
    # nltk.download('stopwords')
    stop = stopwords.words('english')
    text_tokens = word_tokenize(user_input)

    text_tokens_clean = [word for word in text_tokens if not word in stop]
    user_input = (" ").join(text_tokens_clean)

    # # lemmatization using spacy
    nlp = spacy.load("Dataset/en_core_web_lg/en_core_web_lg-3.5.0")  # load an existing English template
    tokens = nlp(user_input)
    lemmas=[]
    for token in tokens:
        lemmas.append(token.lemma_)
    user_input=" ".join(lemmas)

    print("After Text Preprocessing = ", user_input)

    user_input = vectorizer.transform([user_input])
    prediction = classifier.predict_proba(user_input)
    print("[N = %0.6f" % prediction[0][0], ", P = %0.6f]" % prediction[0][1])
    return prediction

if __name__ == '__main__':
    classifier, vectorizer = loadmodel()
    print("[Type Exit to stop program]")
    user_input = input("Write a Review = ")
    while user_input.lower() != "exit":
        prediction = sentimentAnalyser(classifier=classifier,vectorizer=vectorizer,user_input=user_input)
        if prediction[0][0] > prediction[0][1]:
            print("Likely to be Negative")
        else:
            print("Likely to be Positive")
        user_input = input("Write another review = ")
        continue