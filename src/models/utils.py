import json

def rerank(rankings_list, weights=None):
    '''
    Reranks a list of rankings by averaging the ranks of each candidate in the rankings.
    Optionally, weights can be provided to give more importance to some rankings. The weight
    should be a positive (excluding zero) real number. The lower the weight, the more important the ranking.
    Note that the rankings in the list must have the same length and the same candidates in them.

    Args:
        rankings_list: list of rankings to be reranked
        weights: list of weights for each ranking in the list. If None, all rankings are considered
                    equally important.
    Returns:
        final_ranking: list of reranked candidates
    '''
    # making sure all rankings in the list have the same length
    iterator = iter(rankings_list)
    length = len(next(iterator))
    assert all(len(l) == length for l in iterator)
    if weights is not None:
        assert len(rankings_list) == len(weights)

    final_ranking = []
    for i in range(len(rankings_list[0])): # for each query
        reranked_candidates = dict()
        for j in range(len(rankings_list)): # for each ranking
            for cand in rankings_list[j][i]:
                if weights is not None:
                    if cand not in reranked_candidates.keys():
                        reranked_candidates[cand] = weights[j] * rankings_list[j][i].index(cand)
                    else:
                        reranked_candidates[cand] += weights[j] * rankings_list[j][i].index(cand)
                else:
                    if cand not in reranked_candidates.keys():
                        reranked_candidates[cand] = rankings_list[j][i].index(cand)
                    else:
                        reranked_candidates[cand] += rankings_list[j][i].index(cand)
        reranked_list = [key for key in dict(sorted(reranked_candidates.items(), key=lambda item: item[1]/len(rankings_list))).keys()] # sort by average rank
        final_ranking.append(reranked_list)
    
    return final_ranking

def process_predictions(corpus, queries, ranking):
    '''
    Retrieves the predictions in the format required by the evaluation script.

    Args:
        corpus: dataframe with the corpus
        queries: dataframe with the queries
        ranking: list of rankings
    Returns:
        predictions: list of dictionaries with the predictions
    '''
    predictions = []
    for i in range(len(queries)):
        predictions.append(
            {
                "query_id": int(queries.iloc[i]["query_id"]),
                "relevant_candidates": [
                    int(corpus.iloc[int(j)]["argument_id"]) 
                    for j in ranking[i]
                    ]
            }
        )
    return predictions

def parse_sociodemographic_prediction(prediction, legal_values=None):
    '''Parses the prediction of the sociodemographics of an author.
    
    Args:
        prediction: string with the prediction
        legal_values: dictionary with the possible values for each
                      sociodemographic attribute
    Returns:
        prediction_dict: dictionary with the predicted values for each
                         sociodemographic attribute
    '''
    if prediction == None:
        return None
    if legal_values is not None: # use legal values to parse the prediction
            prediction_dict = {}
            for key, value in legal_values.items():
                for val in value:
                    if val in prediction:
                        if key not in prediction_dict:
                            prediction_dict[key] = [val]
                        else:
                            prediction_dict[key].append(val)
    else:
        try:
            return json.loads(prediction)
        except: # very hacky but does the  job ;)
            if '{' in prediction:
                prediction_dict = {}
                pred = prediction.split("{")[1].strip()
                # Remove leading and trailing curly braces
                pred = pred[1:].replace('}', '')
                # Split the string by commas
                pred = pred.split(',')
                # Re-merge if a value starts with a [ and the next value ends with a ]
                for i in range(len(pred)):
                    if pred[i].startswith('[') and pred[i+1].endswith(']'):
                        print(pred[i] + ',' + pred[i+1])
                        pred[i] = pred[i] + ',' + pred[i+1]
                        pred.pop(i+1)
                for p in pred:
                    # Remove leading and trailing whitespace, quotes, and brackets
                    p = p.strip().replace('"', '').replace('[', ''). \
                        replace(']', '').replace("'", '')
                    if ':' in p:
                        # Split the string by colons to retrieve the key and value
                        # per socio-demographic attribute
                        out = p.split(':')
                        # Add the key and value to the dictionary; everything after
                        # a potential second colon is removed as it is not part of
                        # the expected output but rather a result of the LLM output
                        # not adhereing to the expected format.
                        # The value is split by " or " to separate multiple values.
                        prediction_dict[out[0]] = out[1].strip(). \
                            replace(" OR ", " or ").split(" or ")
                        # cleaning up the values, removing parts of the string that
                        # are not part of the expected output
                        for key, value in prediction_dict.items():
                            for i in range(len(value)):
                                value[i] = value[i].split("\n")[0]. \
                                    split(". ")[0].strip()
            else:
                return None
    return prediction_dict