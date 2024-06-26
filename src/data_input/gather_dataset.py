import os
import sys
import operator

import pandas as pd
import pickle
import numpy as np
import readability
import spacy

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_input.load_data import load_corpus, load_queries

def pos_features(text):
    doc = nlp(text)
    doc_pos = [token.pos_ for token in doc]
    doc_length = len(doc_pos)
    counts = {tag: doc_pos.count(tag)/doc_length for tag in upos_tags} # proportion of pos tags in text
    return counts


def entity_feature(text):
    doc = nlp(text)
    doc_ent = [token.ent_iob_ for token in doc]
    doc_length = len(doc_ent)
    count = (doc_ent.count('B')+doc_ent.count('I'))/doc_length # proportion of entities in text
    return count


def morph_features(text):
    doc = nlp(text)
    doc_morph = [token.morph for token in doc]
    doc_length = len(doc_morph)
    tense = sum([1 for token in doc_morph if "Tense=Pres" in token])/doc_length
    mood = sum([1 for token in doc_morph if "Mood=Imp" in token])/doc_length
    person = sum([1 for token in doc_morph if "Person=1" in token])/doc_length
    return {"past_tense": tense, "imperative": mood, "first_person": person}

if __name__=='__main__':
    print("Loading data...")
    # Building training/validation datasets from union of evaluation cycle datasets
    corpus1 = load_corpus("../../data")
    corpus2 = load_corpus("../../data/test_cycle_2")
    corpus3 = load_corpus("../../data/2023-surprise-data")
    corpus = pd.concat([corpus1, corpus2, corpus3])

    queries_train_1 = load_queries("../../data/", "perspective", "train")
    queries_train_2 = load_queries("../../data/test_cycle_2", "perspective", "train")
    queries_train_3 = load_queries("../../data/2023-surprise-data", "perspective", "train")
    queries_train = pd.concat([queries_train_1, queries_train_2, queries_train_3]).drop_duplicates(subset="query_id").reset_index()
    queries_dev_1 = load_queries("../../data", "perspective", "dev")
    queries_dev_2 = load_queries("../../data/test_cycle_2", "perspective", "dev")
    queries_dev_3 = load_queries("../../data/2023-surprise-data", "perspective", "dev")
    queries_dev = pd.concat([queries_dev_1, queries_dev_2, queries_dev_3]).drop_duplicates(subset="query_id").reset_index()

    queries_train_bl1 = load_queries("../../data/", "baseline", "train")
    queries_train_bl2 = load_queries("../../data/test_cycle_2", "baseline", "train")
    queries_train_bl3 = load_queries("../../data/2023-surprise-data", "baseline", "train")
    queries_train_bl = pd.concat([queries_train_bl1, queries_train_bl2, queries_train_bl3]).drop_duplicates(subset="query_id").reset_index()
    queries_dev_bl1 = load_queries("../../data", "baseline", "dev")
    queries_dev_bl2 = load_queries("../../data/test_cycle_2", "baseline", "dev")
    queries_dev_bl3 = load_queries("../../data/2023-surprise-data", "baseline", "dev")
    queries_dev_bl = pd.concat([queries_dev_bl1, queries_dev_bl2, queries_dev_bl3]).drop_duplicates(subset="query_id").reset_index()

    print("Gathering Features...")
    corpus['FleschReadingEase'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['readability grades']['FleschReadingEase'])
    corpus['GunningFogIndex'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['readability grades']['GunningFogIndex'])

    corpus['characters_per_word'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['sentence info']['characters_per_word'])
    corpus['words_per_sentence'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['sentence info']['words_per_sentence'])
    corpus['type_token_ratio'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['sentence info']['type_token_ratio'])
    corpus['long_words'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['sentence info']['long_words'])
    corpus['complex_words'] = corpus['argument'].apply(lambda x: readability.getmeasures(x, lang='de')['sentence info']['complex_words'])

    upos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 
                 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    nlp = spacy.load("de_core_news_sm")

    # add column with pos tags (as dict)
    corpus['POS'] = corpus['argument'].apply(lambda x: pos_features(x))
    # pos dict to single columns
    corpus = pd.concat([corpus, corpus['POS'].apply(pd.Series)], axis=1)
    corpus = corpus.drop('POS', axis=1)

    # add column with entity ratio
    corpus['Entities'] = corpus['argument'].apply(lambda x: entity_feature(x))

    # add column with morphology features
    corpus['Morph'] = corpus['argument'].apply(lambda x: morph_features(x))
    # morph dict to single columns
    corpus = pd.concat([corpus, corpus['Morph'].apply(pd.Series)], axis=1)
    corpus = corpus.drop('Morph', axis=1)

    # add binary stance column as feature
    corpus['stance_num'] = corpus.stance.apply(lambda x: 1 if x == "FAVOR" else 0)

    # Building list of all possible groups and dict of groups per attribute
    attr_val = {}
    all_vals = []
    for attribute in ['gender', 'age',
           'residence', 'civil_status', 'denomination', 'education',
           'political_spectrum']:
        if attribute == 'age':
            attr_val["age_bin"] = list(corpus[f"{attribute}"].unique())
            all_vals += attr_val["age_bin"]
        else:
            attr_val[attribute] = list(corpus[f"{attribute}"].unique())
            all_vals += attr_val[attribute]
    vals = []
    for i, row in corpus.iterrows():
        vals += row["important_political_issues"]
    vals = list(set(vals))
    all_vals += vals
    attr_val["important_political_issue"] = list(vals)

    # building dict for one-hot encoding of groups
    enc_dict = {}
    i = 0
    for attribute in attr_val.keys():
        enc_dict[attribute] = {}
        for value in attr_val[attribute]:
            enc = [0 for _ in range(len(all_vals))]
            enc[i] = 1
            i+=1
            enc_dict[attribute][value] = enc

    train_sd_encoding = []
    train_texts = []
    train_additional_feats = []
    train_labels = []

    val_sd_encoding = []
    val_texts = []
    val_additional_feats = []
    val_labels = []

    train = "./train.tsv"
    with open(train, "w+") as f:
        f.write("encoding\ttext\tadditional_feats\tlabel\n")
    validate = ("./val.tsv")
    with open(validate, "w+") as f:
        f.write("encoding\ttext\tadditional_feats\tlabel\n")

    for attribute in enc_dict.keys():
        train_queries = queries_train[queries_train.demographic_property.apply(lambda x: list(x.keys())[0] == attribute)]
        val_queries = queries_dev[queries_dev.demographic_property.apply(lambda x: list(x.keys())[0] == attribute)]

        print(attribute)

        for val in enc_dict[attribute].keys():
            print(val)
            for _, query in train_queries.iterrows():
                query_text = query["text"]
                # positive argument IDs are the relevant arguments of the query
                positive_ids = query["relevant_candidates"]
                # gather as many negative IDs from all non-relevant argument IDs
                all_cands = queries_train_bl[queries_train_bl["text"] == query["text"]]["relevant_candidates"].values[0]
                # potential negative IDs are all IDs not in the relevant candidates
                negative_ids = list(set(all_cands) - set(positive_ids))[:len(positive_ids)]
                
                with open(train, "a+") as f:
                    for idx in positive_ids:
                        # group one-hot encoding
                        f.write(str(enc_dict[attribute][val]))
                        f.write("\t")
                        # Query and Argument text 
                        text = query["text"] + " " + corpus[corpus["argument_id"] == idx]["argument"].values[0]
                        f.write(text.replace("\n", " "))
                        f.write("\t")
                        # Features
                        f.write(str(list(corpus[corpus["argument_id"] == idx].reset_index().iloc[0][['FleschReadingEase', 'GunningFogIndex', 'characters_per_word', 'words_per_sentence', 'type_token_ratio', 'long_words', 'complex_words', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'Entities', 'past_tense', 'imperative', 'first_person', 'stance_num']])))
                        f.write("\t")
                        # Label
                        f.write("1")
                        f.write("\n")

                    for idx in negative_ids:
                        f.write(str(enc_dict[attribute][val]))
                        f.write("\t")
                        text = query["text"] + " " + corpus[corpus["argument_id"] == idx]["argument"].values[0]
                        f.write(text.replace("\n", " "))
                        f.write("\t")
                        f.write(str(list(corpus[corpus["argument_id"] == idx].reset_index().iloc[0][['FleschReadingEase', 'GunningFogIndex', 'characters_per_word', 'words_per_sentence', 'type_token_ratio', 'long_words', 'complex_words', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'Entities', 'past_tense', 'imperative', 'first_person', 'stance_num']])))
                        f.write("\t")
                        f.write("0")
                        f.write("\n")
            print("Wrote training data")
            
            for _, query in val_queries.iterrows():
                query_text = query["text"]
                # positive argument ids
                positive_ids = query["relevant_candidates"]
                # negative ids
                all_cands = queries_dev_bl[queries_dev_bl["text"] == query["text"]]["relevant_candidates"].values[0]
                negative_ids = list(set(all_cands) - set(positive_ids))[:len(positive_ids)]
                with open(validate, "a+") as f:
                    for idx in positive_ids:
                        f.write(str(enc_dict[attribute][val]))
                        f.write("\t")
                        text = query["text"] + " " + corpus[corpus["argument_id"] == idx]["argument"].values[0]
                        f.write(text.replace("\n", " "))
                        f.write("\t")
                        f.write(str(list(corpus[corpus["argument_id"] == idx].reset_index().iloc[0][['FleschReadingEase', 'GunningFogIndex', 'characters_per_word', 'words_per_sentence', 'type_token_ratio', 'long_words', 'complex_words', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'Entities', 'past_tense', 'imperative', 'first_person', 'stance_num']])))
                        f.write("\t")
                        f.write("1")
                        f.write("\n")
                    for idx in negative_ids:
                        f.write(str(enc_dict[attribute][val]))
                        f.write("\t")
                        text = query["text"] + " " + corpus[corpus["argument_id"] == idx]["argument"].values[0]
                        f.write(text.replace("\n", " "))
                        f.write("\t")
                        f.write(str(list(corpus[corpus["argument_id"] == idx].reset_index().iloc[0][['FleschReadingEase', 'GunningFogIndex', 'characters_per_word', 'words_per_sentence', 'type_token_ratio', 'long_words', 'complex_words', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'Entities', 'past_tense', 'imperative', 'first_person', 'stance_num']])))
                        f.write("\t")
                        f.write("0")
                        f.write("\n")
            print("Wrote validation data")

