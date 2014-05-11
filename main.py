__author__ = 'saswatac'

import StemmedVectorizer
import sklearn.datasets
import nltk.stem
import scipy as sp
from sklearn.cluster import KMeans
import numpy as np

def main():

    MLCOMP_DIR = r"C:\Users\Public\Documents\courses\ML\newsGroupDataSet"
    groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']
    train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR, categories=groups)
    stemmer = nltk.stem.SnowballStemmer("english")
    vectorizer = StemmedVectorizer.StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', charset_error='ignore', stemmer=stemmer)
    vectorized = vectorizer.fit_transform((open(f).read() for f in train_data.filenames))
    print vectorized[1,:].todense()
    num_clusters = 50
    km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
    km.fit(vectorized)
    labels = np.array(km.labels_)
    print labels
    for i in range(0,num_clusters):
        num_docs_in_cluster = sum((labels == i))
        print "cluster ",i,": ", "number of documents ",num_docs_in_cluster

    new_post = "Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks."
    new_post_vec = vectorizer.transform([new_post])
    print new_post_vec.todense()
    new_post_label = km.predict(new_post_vec)[0]
    similar_indices = (labels==new_post_label).nonzero()[0]
    similar = []
    for i in similar_indices:
        dist = sp.linalg.norm((new_post_vec-vectorized[i,:]).toarray())
        similar.append((dist, train_data.filenames[i]))

    similar = sorted(similar)
    print "similar posts: ",len(similar)
    printfile(similar[0][1])
    printfile(similar[-1][1])

def printfile(fname):
    contents = open(fname).read()
    print contents

if __name__ == '__main__':
    main()
