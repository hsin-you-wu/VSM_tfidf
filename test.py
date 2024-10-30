from helper import *
from vs import *

# load docs
engNews = 'EnglishNews'
documents = getDoc(engNews)
print("##### Task 1 #####")
query = input("Query: ")

vs_tf = VectorSpace(documents, "tf", query)
vs_tfidf = VectorSpace(documents, "tfidf", query)

# Q1-1
print("----- TF Cosine -----")
vs_tf.print_top10("cosine")

# Q1-2
print("----- TF-IDF Cosine -----")
vs_tfidf.print_top10("cosine")

# Q1-3
print("----- TF Euclidean -----")
vs_tf.print_top10("euclidean")

# Q1-4
print("----- TF-IDF Euclidean -----")
vs_tfidf.print_top10("euclidean")