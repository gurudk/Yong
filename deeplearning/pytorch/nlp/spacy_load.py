import spacy
import os
from spacy.lang.de.examples import sentences

try:
    spacy_de = spacy.load("de_core_news_sm")
except IOError:
    os.system("python -m spacy download de_core_news_sm")
    spacy_de = spacy.load("de_core_news_sm")

try:
    spacy_en = spacy.load("en_core_web_sm")
except IOError:
    os.system("python -m spacy download en_core_web_sm")
    spacy_en = spacy.load("en_core_web_sm")

nlp = spacy.load("de_core_news_sm")
doc = nlp(sentences[1])
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)
