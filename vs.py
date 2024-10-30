from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from helper import *
import jieba
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect


class VectorSpace():
    """
    - build document vectors & query vector
    - calculate similarity of queryVector to each documentVector
    - print top 10 results
    """
    
    def __init__(self, documents: dict, weighting_method: str, query: list):
        self.documents = documents
        self.documentVectors = {}
        self.queryVector = []
        self.corpus = []

        # build vectors
        self.buildVectors(weighting_method, query)
        
    
    def clean(self, document: str) -> list:
        " clean a document. only deal with english "

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        cleaned_tokens = []

        tokens = word_tokenize(document) # tokenize
        for token in tokens:
            token = token.lower()  # Lowercasing
            if token.isalnum() and token not in stop_words:  # Remove punctuation and stopwords
                token = lemmatizer.lemmatize(token)  # Lemmatize
                cleaned_tokens.append(token)

        cleaned_document = " ".join(cleaned_tokens)
        return cleaned_document


    def buildVectors(self, weighting_method, query) -> None:
        """ Given the weighting method, create a vector for each document and store it in self.documentVectors. 
            Also creates the queryVector.
        """
        
        # create a vectorizer object
        if weighting_method == 'tfidf':
            vectorizer = TfidfVectorizer(norm=None)
        elif weighting_method == 'tf':
            vectorizer = CountVectorizer()
        else:
            raise ValueError(f"Unsupported weighting method: {weighting_method}")
        
        # clean documents and query
        cleaned_documents = {filename: self.clean(content) for filename, content in self.documents.items()}
        cleaned_query = self.clean(query)

        # fit the vectorizer
        all_texts = list(cleaned_documents.values()) + [cleaned_query]
        vectorizer.fit(all_texts)

        # store the corpus
        self.corpus = vectorizer.get_feature_names_out()
        
        # for each document, create a documentVector and store in dict
        document_vectors = vectorizer.transform(list(cleaned_documents.values()))
        for i, filename in enumerate(cleaned_documents.keys()):
            self.documentVectors[filename] = document_vectors[i].toarray().flatten()

        # create a queryVector
        self.queryVector = vectorizer.transform([cleaned_query]).toarray().flatten()


    def calculateSimilarities(self, similarity_method, queryVector=None) -> dict:
        """ - calculate the similarity between every documentVector & the queryVector. 
            - returns a dict of {filename: similarity score}.
            - if a queryVector is given, it calculates based on that (for PRF).
        """
        results = {}
        if queryVector is None:
            queryVector = self.queryVector

        if similarity_method == "cosine":
            for filename in self.documentVectors:
                score = cosine(self.documentVectors[filename], queryVector)
                results[filename] = score
        if similarity_method == "euclidean":
            for filename in self.documentVectors:                
                score = euclidean(self.documentVectors[filename], queryVector)
                results[filename] = score
        
        return results


    def get_top10(self, similarity_method: str) -> dict:
        """ get the top 10 results. """
        # get top 10 results
        results = self.calculateSimilarities(similarity_method)
        if similarity_method == 'cosine':
            sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        if similarity_method == 'euclidean':
            sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=False)

        top10 = {item[0]: item[1] for item in sorted_results[:10]}
        return top10
    
    
    def print_top10(self, similarity_method, top10=None):
        """ print results in desired format. """
        if not top10:
            top10 = self.get_top10(similarity_method)

        # Print headers
        print(f"{'NewsID':<20} {'Score':<10}")
        
        # Print the sorted results with rounding
        for filename, score in top10.items():
            print(f"{filename:<20} {score:.5f}")

        print()


class VectorSpace_ch(VectorSpace):
    
    def clean(self, document: str) -> list:
        """ Clean a single document. Able to deal with ch and en. """

        # Set up English-specific tools
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        cleaned_tokens = []

        # Split document into chunks of Chinese and English
        chunks = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\u4e00-\u9fff\s]+', document)

        for chunk in chunks:
            try:
                language = detect(chunk)  
            except:
                continue 
            
            if language == 'en':
                tokens = word_tokenize(chunk) # tokenize
                for token in tokens:
                    token = token.lower()  # Lowercasing
                    if token.isalnum() and token not in stop_words:  # Remove punctuation and stopwords
                        token = lemmatizer.lemmatize(token)  # Lemmatize
                        cleaned_tokens.append(token)

            else:
                chinese_tokens = jieba.cut(chunk) # tokenize
                for token in chinese_tokens:
                    if token.isalnum():  # Remove punctuation
                        cleaned_tokens.append(token)

        cleaned_document = " ".join(cleaned_tokens)
        return cleaned_document


def buildFeedbackQueryVector(top_result_filename: str, query: str, documents: dict, corpus: list):
    """ Build feedback-query-vector based on the top retrieved result. """
    # extract verbs and nouns
    top_result_content = documents[top_result_filename]
    tokens = word_tokenize(top_result_content)
    pos_tags = pos_tag(tokens)
    cleaned_feedback =  [word for word, pos in pos_tags if pos.startswith('VB') or pos.startswith('NN')]
    cleaned_feedback = " ".join(cleaned_feedback)
    
    # create a vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus)

    # build queryVector and feedbackVector
    queryVector = vectorizer.transform([query]).toarray().flatten()
    feedbackVector = vectorizer.transform([cleaned_feedback]).toarray().flatten()
    
    # calculate fqVector
    fqVector = queryVector + (0.5 * feedbackVector)
    return fqVector
    

def rerank(vs: VectorSpace, similarity_method: str, query, documents: dict):
    """ Performs the PRF and prints the results. """
    
    # build fqVector
    top_result_filename = list(vs.get_top10(similarity_method).keys())[0]
    fqVector = buildFeedbackQueryVector(top_result_filename, query, documents, vs.corpus)

    # calculate similarity and get the top10 results
    results = vs.calculateSimilarities(similarity_method, fqVector)
    if similarity_method == 'cosine':
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    if similarity_method == 'euclidean':
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=False)
    top10 = {item[0]: item[1] for item in sorted_results[:10]}

    # print results
    vs.print_top10(similarity_method, top10)
