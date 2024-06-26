import os

import pandas as pd

def preprocess_corpus(corpus):
    '''Preprocess the corpus.
    Args:
        corpus (pd.DataFrame): DataFrame with the following columns:
            argument_id (str): Unique identifier for the argument.
            argument (str): The argument text.
            stance (str): The stance of the argument. 'FAVOR' or 'AGAINST'.
            topic (str): The topic of the argument.
                         One of the following:
                            'Immigration',
                            'Society',
                            'Welfare',
                            'Education',
                            'Finances', 
                            'Economy',
                            'Security',
                            'Healthcare',
                            'Infrastructure & Environment',
                            'Digitisation'
            demographic_profile (dict): Dictionary with the following keys:
                gender (str): Gender of the author.
                age (str): Age group of the author.
                residence (str): Residence of the author.
                                 Rural (Land) or Urban (Stadt).
                civil_status (str): Civil status of the author.
                denomination (str): Religious denomination of
                                    the author.
                education (str): Education level of the author.
                political_spectrum (str): Political orientation
                                          of the author.
                important_political_issues (list): Important political
                                                   issues for the author.
    Returns:
        pd.DataFrame: DataFrame with columns mentioned above and additional
                      columns for each socio-cultural attribute, and splits
                      political spectrum into rile and galtan.
    '''
    # Flatten the demographic_profile column, makes individual
    # socio-cultural attributes accessible as columns;
    # only valid since all of the documents have the same attributes.
    for key in corpus.demographic_profile[0].keys():
        corpus[f"{key}"] = corpus.demographic_profile.\
            apply(lambda x: x[f"{key}"])
    
    # Split the political_spectrum into rile and galtan.
    corpus[["rile", "galtan"]] = corpus.political_spectrum.str.split(
        " und ", expand=True
        )
        
    # Drop the original demographic_profile column.
    corpus = corpus.drop(columns=['demographic_profile'])

    return corpus

def load_corpus(datapath):
    '''Load and preprocess the corpus from the data directory.
    Assumes the following folder structure:
    [datapath]/corpus.jsonl
    Args:
        datapath (str): Path to the data directory.
    Returns:
        pd.DataFrame: DataFrame with the following columns:
            argument_id (str): Unique identifier for the argument.
            argument (str): The argument text.
            stance (str): The stance of the argument. 'FAVOR' or 'AGAINST'.
            topic (str): The topic of the argument.
                         One of the following:
                            'Immigration',
                            'Society',
                            'Welfare',
                            'Education',
                            'Finances', 
                            'Economy',
                            'Security',
                            'Healthcare',
                            'Infrastructure & Environment',
                            'Digitisation'
            gender (str): Gender of the author.
            age (str): Age group of the author.
            residence (str): Residence of the author.
                             Rural (Land) or Urban (Stadt).
            civil_status (str): Civil status of the author.
            denomination (str): Religious denomination of
                                the author.
            education (str): Education level of the author.
            political_spectrum (str): Political orientation
                                      of the author.
            important_political_issues (list): Important political
                                               issues for the author.
    '''
    corpus = pd.read_json(os.path.join(datapath, 'corpus.jsonl'), lines=True)
    corpus = preprocess_corpus(corpus)
    return corpus

def load_queries(datapath, scenario, split):
    '''Load queries from a given scenario and split.
    Assumes folder structure as follows:
    [datapath]/[scenario]_queries/queries_[split].jsonl
    Args:
        datapath (str): Path to the data directory.
        scenario (str): Either baseline or perspective.
        split (str): train, dev, or test.
    Returns:
        pd.DataFrame: DataFrame with the following columns:
            query_id (str): Unique identifier for the query.
            text (str): The query text.
            demographic_property (dict): Dictionary with the demographic
                                         property of interest.
            relevant_candidates (list): List of relevant candidate arguments
                                        (their IDs).
    '''
    queries = pd.read_json(os.path.join(datapath, f'{scenario}-queries/'
                                   f'queries_{split}.jsonl'), lines=True)
    return queries

