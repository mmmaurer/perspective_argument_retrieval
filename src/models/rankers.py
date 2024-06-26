import string

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from spacy.lang.de.stop_words import STOP_WORDS as stop_words_de
from spacy.lang.fr.stop_words import STOP_WORDS as stop_words_fr
from spacy.lang.it.stop_words import STOP_WORDS as stop_words_it

# Edit the stop words for the languages as needed
STOP_WORDS = {
    'stop_words_de': stop_words_de,
    'stop_words_fr': stop_words_fr,
    'stop_words_it': stop_words_it
}

class TfIdfRanker():
    ''' Ranker based on tf-idf representations.
    '''
    def __init__(self, languages=['de', 'fr', 'it']):
        ''' Initialize the ranker.

        Args:
            languages: list of strings. Languages to use for stop words.
        '''
        self.languages = languages
        self.stop_words = self.get_stop_words()
        self.vectorizer = TfidfVectorizer()
        self.corpus = None
        self.tfidf_matrix = None

    def fit_trainsform(self, corpus):
        ''' Fit the vectorizer and transform the corpus.
        
        Args:
            corpus: list of strings.
        '''
        self.corpus = corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def transform(self, sentences):
        ''' Transform the sentences into tf-idf vectors.
        
        Args:
            sentences: list of strings.
        '''
        return self.vectorizer.transform(sentences)

    def get_stop_words(self):
        ''' Get the stop words for the selected languages.
        '''
        stop_words = set()
        for lang in self.languages:
            stop_words.update(STOP_WORDS[f'stop_words_{lang}'])
        return stop_words

    def rank(self, 
             queries,
             preselected_ids=None,
             top_k=40,
             metric='euclidean'):
        ''' Rank the queries based on the tf-idf representations.
        Note that currently, using a list of preselected_ids is not
        supported if the list of queries contains more than one element.
        Essentially, you can only process all the queries at once if you
        do not provide a list of preselected_ids at the moment.  

        Args:
            queries: list of strings. Queries to rank.
            preselected_ids: list of integers. Indices of the documents
                to consider for ranking.
            top_k: integer. Number of documents to return.
            metric: string. Metric to use for ranking. Default is 'euclidean'.
                            Options are the same as for the sklearn
                            pairwise_distances. Please consult their
                            documentation for more information.
        '''
        queries_preprocessed = [self._preprocess(query)
                                for query in queries]
        query_vector = self.vectorizer.transform(queries_preprocessed)
        if preselected_ids is not None:
            distances = pairwise_distances(query_vector,
                                           self.tfidf_matrix[preselected_ids],
                                           metric=metric)
        else:
            distances = pairwise_distances(query_vector,
                                        self.tfidf_matrix,
                                        metric=metric)
        res = [distances[i].argsort()[:top_k] for i in range(len(queries))]
        if preselected_ids is not None:
            # return original indices (from preselected_ids)
            res = [[preselected_ids[int(i)] for i in r] for r in res] 
        else:
            res = [[int(i) for i in r] for r in res] # convert to list of integers
        return res
    
    def _preprocess(self, text):
        return " ".join(token if token not in self.stop_words 
                        else "" for token in text.lower(). \
                            translate(str.maketrans('', '', 
                                                    string.punctuation)
                                                    ).split())

class SentenceTransformerRanker():
    ''' Ranker based on bi-encoder transformer sentence embeddings.
    '''
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        ''' Initialize the ranker.

        Args:
            model_name: string. Name of the model to use for embeddings.
                                Alternatively, you can provide the path to
                                your own model. Default is
                                'paraphrase-multilingual-mpnet-base-v2'.
        '''
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus = None
        self.corpus_embeddings = None

    def fit_trainsform(self, corpus, show_progress_bar=True):
        ''' Fit the model and transform the corpus.

        Args:
            corpus: list of strings.
            show_progress_bar: boolean. Whether to show the progress bar
                                    during the encoding process.
        '''
        self.corpus = corpus
        self.corpus_embeddings = self.model.encode(corpus, 
                                                   show_progress_bar=show_progress_bar)

    def transform(self, sentences):
        ''' Transform the sentences into embeddings.

        Args:
            sentences: list of strings.
        '''
        return self.model.encode(sentences)

    def rank(self, 
             queries,
             preselected_ids=None,
             top_k=40):
        ''' Rank the queries based on the sentence embeddings.
        Note that currently, using a list of preselected_ids is not
        supported if the list of queries contains more than one element.
        Essentially, you can only process all the queries at once if you
        do not provide a list of preselected_ids at the moment.

        Args:
            queries: list of strings. Queries to rank.
            preselected_ids: list of integers. Indices of the documents
                to consider for ranking.
            top_k: integer. Number of documents to return.
        '''
        query_embeddings = self.model.encode(queries)
        if preselected_ids is not None:
            if query_embeddings is not None:
                similarities = util.cos_sim(query_embeddings,
                                            self.corpus_embeddings[
                                                preselected_ids])
            else: # if there is no representation for the queries, 
                  # just return preselected_ids
                return preselected_ids
        else:
            similarities = util.cos_sim(query_embeddings,
                                        self.corpus_embeddings)
        
        res = [similarities[i].argsort(descending=True)[:top_k] for \
               i in range(len(queries))]
        if preselected_ids is not None:
            # return original indices (from preselected_ids)
            res = [[preselected_ids[int(i)] for i in r] for r in res]
        else:
            res = [[int(i) for i in r] for r in res] # convert to list of integers
        return res
    