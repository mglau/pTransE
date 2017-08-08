from gensim.corpora import WikiCorpus
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
inp = os.path.join(dir_path, 'enwiki-latest-pages-articles10.xml-p2336425p3046511.bz2')
outp = os.path.join(dir_path, 'wiki2')
wiki = WikiCorpus(inp, lemmatize=False)
output = open(outp, 'w')
for text in wiki.get_texts():
    output.write(b' '.join(text).decode('utf-8') + '\n')
output.close()