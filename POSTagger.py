from pathlib import Path
import spacy
from spacy import displacy

# SoLA stands for Software-defined Language Analyzer

# POS Tagging, Sentiment Analysis and Named Entity Recognition(NER) based system via Machine Learning(Ml) models
# Contains a Web scraper for quick information retrieval based on text analysis results

# This file deals with POS Tagging using a pre-trained model from spacy

def loadModel():
    return spacy.load("Dataset/en_core_web_lg/en_core_web_lg-3.5.0")

def POSTagger(nlp, user_input):
    doc = nlp(user_input)
    print("Parts of Speech Tagging -> ")
    for token in doc:
        print(token, " | ", token.pos_ , " | ", spacy.explain(token.pos_)," | ", token.tag_, " | ", spacy.explain(token.tag_))
    counts = doc.count_by(spacy.attrs.POS)
    print(" Total Counts ->")
    for pos,count in counts.items():
        print(doc.vocab[pos].text, " = ", count)
    svg = displacy.render(doc, style="dep")
    output_path = Path("VisualOutputs/pos.svg")
    output_path.open("w", encoding="utf-8").write(svg)
    with open("templates/pos.html", "w") as f:
        f.write(svg)

if __name__ == '__main__':
    nlp = loadModel()
    print("[Type Exit to stop program]")
    user_input = input("Write a Proper Sentence = ")
    while user_input.lower() != "exit":
        POSTagger(nlp, user_input=user_input)
        user_input = input("Write a Proper Sentence = ")
        continue
