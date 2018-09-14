from tfidf import *
import os

zipfilename = os.path.expanduser(sys.argv[1])
summarizefile = sys.argv[2]

corpus = load_corpus(zipfilename)
tfidf = compute_tfidf(corpus)
tuple_list = summarize(tfidf, corpus[summarizefile], 20)

for i in range(len(tuple_list)):
    print tuple_list[i][0], "%.3f" % tuple_list[i][1]
