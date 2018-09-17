# Overview

The goal of this project is to practice a core technique used in text analysis called *TFIDF* or *term frequency, inverse document frequency*. It uses what is called a *bag-of-words* representation where the order of words in a document don't matter -- we care only about the words and how often they occur. A word's TFIDF value is often used as a feature for document clustering or classification. The more a term helps to distinguish its enclosing document from other documents, the higher its TFIDF score. As such, words with high TFIDF scores are often very good summarizing keywords for document.

As a practical matter, the content of the articles are in real XML files (Reuters articles) and parsed by Python.

It leverages [NLTK (Natural Language Toolkit)](https://www.nltk.org/) to tokenize the words and uses [scikit-learn](http://scikit-learn.org/) 's ENGLISH_STOP_WORDS set for stop words removal.

The supportive Python libraries used are NLTK and Scikit-Learn.
