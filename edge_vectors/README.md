This module learn edge vectors given word vectors (vectors of node labels). We had learned word vectors using a completely independent unsupervised software word2vec. We had used this software on 1 million pubmed articles to generate file "pubmed.bin" at the following path. Having this file in place, the wordvec.py module can load this file and retrieve wordvector for almost any word in bio-text. Since the "pubmed.bin" file is 2.5gb, we are not able to avail it here. Users are recommended to learn it on their own using any source unsupervised-text data they are interested in. 

Our module for generating edge vectors is quite general and not specific to bio-text or word vectors from word2vec software. One can obtain word vectors for nodes in a graph using any software that suits them. This works as long as domain specific words are as node labels and edges are representation specific.  

link for word2vec software 
https://code.google.com/archive/p/word2vec/

path where the word2vec software should put the outfile
./wordvectors/pubmed.bin


AMR and SDG are represented in their formats as per preferences of the respective researchers. Herein, we are relying on a common format for AMR and SDG, i.e. dot format graph. For each single AMR/SDG, there should be a single dot file. Using "pydot" package, our software reads these dot files into our own object format. Also, the software is not restricted to the AMR, SDG representations but useful for any directed graph. 

For a preliminary test of the software, we have add approx. 250 AMR and SDG in dot representation at path below. The python module "edge_labels_propagation.py" loads these dot files and then automatically generates the pickeled files for edge vectors. Here, note that software is run on AMR and SDG independently. There is use of edge vectors in AMReven if one is not using AMR and SDG jointly. Same applies for SDG. The corresponding shell scripts files to generate these edge vectors are at paths mentioned below. 

Note that while the algorithm for learnig edge vectors is fast, preprocessing of dot files takes time. In the future, a wrapper will be added to avoid dotfile representation at least for the case. Also, the linear system is solved as least squares, that is something not explicitly mentioned in the paper. The linear system can solved even faster with regularization, and probably with better numbers.  

subdirectory path dot files data for AMR and SDG:
./amr_sdg_dot_files

Shell scripts:

mylocaljob_propagate_edge_vectors_amr.sh

mylocaljob_propagate_edge_vectors_sdg.sh


For any queries, shoot an email at 
sahilgar@usc.edu

