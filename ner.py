from msilib.schema import File
import spacy
from pathlib import Path
from spacy import displacy
from webscraper import getInfoFromWeb
import webbrowser
# SoLA stands for Software-defined Language Analyzer

# POS Tagging, Sentiment Analysis and Named Entity Recognition(NER) based system via Machine Learning(Ml) models
# Contains a Web scraper for quick information retrieval based on text analysis results

# This file deals with Named Entity Recognization using a pre-trained model from spacy and combines it with web scraping


def loadModel():
    return spacy.load("Dataset/en_core_web_lg/en_core_web_lg-3.5.0")

def namedEntityRecogniser(nlp, user_input):
    doc = nlp(user_input)
    for entities in doc.ents:
        print(entities.text, " | ", entities.label_, " | ", spacy.explain(entities.label_))
    svg = displacy.render(doc, style="ent")
    output_path = Path("VisualOutputs/namedEntities.svg")
    output_path.open("w", encoding="utf-8").write(svg)
    with open("templates/ner.html","w") as f :
        f.write(svg)
    # uncomment the two lines of code below to disable web scraping
    for entities in doc.ents:
        print(entities.text+"<>\n", getInfoFromWeb(topic=entities.text))

if __name__ == '__main__':
    nlp = loadModel()
    print("[Type Exit to stop program]")
    user_input = input("Write a Proper Sentence = ")
    while user_input.lower() != "exit":
        namedEntityRecogniser(nlp=nlp, user_input=user_input)
        user_input = input("Write a Proper Sentence = ")
        continue