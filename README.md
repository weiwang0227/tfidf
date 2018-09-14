# Overview

The goal of this homework is to learn a core technique used in text analysis called *TFIDF* or *term frequency, inverse document frequency*. It uses what is called a *bag-of-words* representation where the order of words in a document don't matter--we care only about the words and how often they occur. A word's TFIDF value is often used as a feature for document clustering or classification. The more a term helps to distinguish its enclosing document from other documents, the higher its TFIDF score. As such, words with high TFIDF scores are often very good summarizing keywords for document.

As a practical matter, the content of the articles are in real XML files (Reuters articles) and parsed by Python.

The supportive Python libraries used are NLTK and Scikit-Learn.
