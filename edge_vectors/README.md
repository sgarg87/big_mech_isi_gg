This module learn edge vectors given word vectors (vectors of node labels). We had learned word vectors using a completely independent unsupervised software word2vec. We had used this software on 1 million pubmed articles to generate file "pubmed.bin" at the following path. Having this file in place, the wordvec.py module can load this file and retrieve wordvector for almost any word in bio-text. Since the "pubmed.bin" file is 2.5gb, we are not able to avail it here. Users are recommended to learn it on their own using any source unsupervised-text data they are interested in. 

Our module for generating edge vectors is quite general and not specific to bio-text or word vectors from word2vec software. One can obtain word vectors for nodes in a graph using any software that suits them. This works as long as domain specific words are as node labels and edges are representation specific.  

link for word2vec software 
https://code.google.com/archive/p/word2vec/

path where the word2vec software should put the outfile
./wordvectors/pubmed.bin
