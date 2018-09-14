from tfidf import *
import sys
from collections import Counter


file_path = os.path.expanduser(sys.argv[1])

with open(file_path, 'r') as file:
    xmltext = file.read()
    tokens = tokenizer(gettext(xmltext))
    token_counters = Counter(tokens)
    common_number = 10 if len(token_counters) > 10 else len(token_counters)
    output_list = token_counters.most_common(common_number)
    for element in output_list:
        print element[0], element[1]