# Project Setup  
  First, unzip ```Documents.7z```. Make sure the documents are stored in a folder called ```Documents```.
  Execute the following two CLIs to download the required resources:  
  ```pip install -r requirements.txt```  
  ```python3 download_resources.py```  

# Usage
  Execute ```python3 main.py --task {task num from 1~4}``` to see task results.  
  For example, ```python3 main.py --task 1``` will return the results for task 1. 
  
# Tasks  

### Task 1  
The VSM is able to retrieve the relevant news to the given query from a set of 7,875 English news collected from reuters.com according to different weighting schemes and similarity metrics. There are 4 kinds of combinations:  

  - TF weighting + Cosine Similarity
  - TF-IDF weighting + Cosine Similarity
  - TF weighting + Euclidean Distance
  - TF-IDF weighting + Euclidean Distance

### Task 2  
Relevance Feedback is an IR technique for improving retrieved results. The simplest approach is Pseudo Feedback, the idea of which is to feed the results retrieved by the given query, and then to use the content of the fed results as supplement queries to re-score the documents.  
For this task, we will use the Nouns and Verbs from the retrieved results from task 1 to perform Pseudo Feedback, and return a new retreival result. 

### Task 3  
Task 3 performs similariy as task 1, only that it uses a different dataset -- a set of 2,589 news containing both English and Chinese. There are 2 kinds of combinations:  
  
  -  TF weighting + Cosine Similarity
  -  TF-IDF weighting + Cosine Similarity

### Task 4  
This task focus on a smaller dataset of 1,460 documents, 76 queries and their labelled relevant documents. It then calculates the following metrics:  

  - MRR@10
  - MAP@10
  - Recall@10

# Reminder  
  Task 1 and 2 should take less than 30 seconds to execute.  
  Task 3 tends to take longer for the results to show, usually taking about 2 to 3 minutes.  
  Task 4 takes about 1 to 2 minutes.
