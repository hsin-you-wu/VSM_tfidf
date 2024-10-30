import glob
import numpy as np
import os
import pickle
import csv
import ast
import re


def save_documents(documents, filename='documents.pkl'):
    # Ensure the resources directory exists
    os.makedirs('resources', exist_ok=True)
    filepath = os.path.join('resources', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(documents, f)


def load_documents(filename='documents.pkl'):
    filepath = os.path.join('resources', filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def getDoc(filename):
    """ load documents or read from folder if one doesn't exist. """
    documents = load_documents(f'{filename}.pkl')
    if documents is None:
        folder_path = f'Documents/{filename}'
        documents = read_txt_files(folder_path)
        save_documents(documents, f'{filename}.pkl')
    return documents


def read_txt_files(folder_path) -> dict:
    """Given a folder path, read all txt files in it and store the filenames and contents into a dict."""

    # find all file paths that ends with .txt
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

    documents = {}

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r') as file:
            documents[file_name] = (file.read())
    
    return documents


def read_tsv_files(file_path) -> dict:
    result_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            query_id = row[0]
            doc_ids = ast.literal_eval(row[1])
            if query_id not in result_dict:
                result_dict[query_id] = []
            result_dict[query_id] = doc_ids
    return result_dict


def cosine(vec1, vec2): 
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Check for zero norms to avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)


def euclidean(vec1, vec2):   
    # Calculate the Euclidean distance using scipy
    distance = np.linalg.norm(vec1 - vec2)
    
    return distance


def extract_id(doc_filename):
    match = re.search(r'\d+', doc_filename)
    return str(match.group()) if match else None


def MRR(results: dict, answers: dict):
    """ calculate the mrr score of results = {queryId: [top 10 results' documentID]}. """
    
    reciprocals = []

    for queryId, top10 in results.items():
        answer = answers.get(queryId, [])
        
        # get the first relevant doc's rank
        rel_rank = None
        for rank, doc_id in enumerate(top10):
            if int(doc_id) in answer:
                rel_rank = rank + 1
                break
        
        # calculate reciprocal
        if rel_rank is not None:
            reciprocals.append(1/ rel_rank)
        else:
            reciprocals.append(0)
    
    # calculate MRR score
    mrr = sum(reciprocals) / len(reciprocals) if reciprocals else 0
    return mrr
        

def AP(top10: list, answer: list):
    """ calculate the Average Precision. """
    precisions = []
    relevant_count = 0

    # calculate precision at each rank
    for rank in range(len(top10)):
        if int(top10[rank]) in answer:
            relevant_count += 1
            precision = relevant_count / (rank + 1)
            precisions.append(precision)
    
    # calculate average precision
    ap = sum(precisions) / relevant_count if precisions else 0
    return ap


def MAP(results: dict, answers: dict):
    """ calculate the mean of all queries' AP. """
    aps = []
    for queryId, top10_ids in results.items():
        ap = AP(top10_ids, answers[queryId])
        aps.append(ap)

    map_ = sum(aps) / len(results) 
    return map_


def RECALL(results: dict, answers: dict):
    recalls = []
    for queryId, top10_ids in results.items():
        relevant_count = 0
        for i in range(len(top10_ids)):
            if int(top10_ids[i]) in answers[queryId]:
                relevant_count += 1
        
        if relevant_count == 0:
            recall = 0
        else:
            recall = relevant_count / len(answers[queryId])
        recalls.append(recall)
    
    average_recall = sum(recalls) / len(recalls)
    return average_recall
