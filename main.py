import argparse
from helper import *
import logging
import jieba
from vs import *


jieba.setLogLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Process tasks.')
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], required=True, help='Task number to execute (1, 2, 3, or 4)')
    args = parser.parse_args()

    if args.task == 1:
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
        
    elif args.task == 2:
        # load docs
        documents = getDoc('EnglishNews')

        query = input("Query: ")
        vs_tfidf = VectorSpace(documents, "tfidf", query)

        print("##### Task 2 #####")
        rerank(vs_tfidf, 'cosine', query, documents)

    elif args.task == 3:
        documents_ch = getDoc('ChineseNews')
        print("##### Task 3 #####")
        query_ch = input("Query: ")

        vs_tf_ch = VectorSpace_ch(documents_ch, "tf", query_ch)
        vs_tfidf_ch = VectorSpace_ch(documents_ch, "tfidf", query_ch)

        # Q3-1
        print("----- TF Cosine -----")
        vs_tf_ch.print_top10("cosine")

        # Q3-2
        print("----- TF-IDF Cosine -----")
        vs_tfidf_ch.print_top10("cosine")

    elif args.task == 4:
        # load resources
        documents = getDoc('collections')
        queries = getDoc('queries')
        answers = read_tsv_files('Documents/rel.tsv')
        results = {}

        # get top10 results for each query
        for queryId, query in queries.items():
            vs_tfidf = VectorSpace(documents, 'tfidf', query)
            top10 = list(vs_tfidf.get_top10('cosine').keys())
            top10_id = [extract_id(doc) for doc in top10]
            results['q' + extract_id(queryId)] = top10_id

        print("##### Task 4 #####")
        print("----- TF-IDF Cosine -----")

        # MRR
        mrr = MRR(results, answers)
        print(f"MRR@10{'':<20} {mrr:.5f}")

        # MAP
        map_ = MAP(results, answers)
        print(f"MAP@10{'':<20} {map_:.5f}")

        # Recall
        recall = RECALL(results, answers)
        print(f"RECALL@10{'':<20} {recall:.5f}")


    else:
        print("Enter a number from 1 ~ 4")


if __name__ == '__main__':
    main()
    
    
    