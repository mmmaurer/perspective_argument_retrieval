from random import sample
from sentence_transformers import InputExample
from tqdm import tqdm

def vanilla_pre_template(sociodemographic_property, value):
    return f'Gegeben {sociodemographic_property}: {value}. '

def retrieve_dataset_instances(
        queries,
        corpus,
        translation_dict=None,
        demographic_properties=True,
        lang='de',
        show_progress_bar=False,
        proportion_negatives=1,
        pre_template=vanilla_pre_template
        ):
    '''Retrieves and processes instances for creating a dataset for fine-tuning and evaluating a 
    sentence-transformer model.

    Args:
        queries (pd.DataFrame): DataFrame containing the queries.
        corpus (pd.DataFrame): DataFrame containing the corpus.
        translation_dict (dict, optional): Dictionary containing translations for the demographic properties.
                                           Defaults to None. If None, the original property names are used.
                                           Structure: {property/value: {lang: translation}}.
        demographic_properties (bool, optional): Whether to include demographic properties in the queries.
                                                 Defaults to True.
        lang (str, optional): Language of the translations. Defaults to 'de'.
        show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to False.
        proportion_negatives (int, optional): Proportion of negative examples to add. Defaults to 1.
                                              0.5 means half as many negative examples as positive examples,
                                              2 means twice as many negative examples as positive examples.
        pre_template (function, optional): Function to generate a prefix for the query.
                                           Defaults to vanilla_pre_template.
    Returns:
        list: List of InputExample instances for fine-tuning the sentence-transformer model.
    '''
    
    def process_query(query, corpus, examples_list, translation_dict, all_arg_ids, pre_template=pre_template):
        '''Processes a single query and adds examples to the examples_list.'''
        sent1 = query.text
        if demographic_properties:
            if translation_dict:
                prop = translation_dict[list(query.demographic_property.keys())[0]][lang]
                val = translation_dict[list(query.demographic_property.values())[0]][lang]
            else:
                prop = list(query.demographic_property.keys())[0]
                val = list(query.demographic_property.values())[0]
            pre = pre_template(prop, val)
            sent1 = pre + sent1
        # adding positive examples
        for arg_id in query.relevant_candidates:
            sent2 = corpus[corpus.argument_id == arg_id].argument.values[0]
            examples_list.append(InputExample(texts=[sent1, sent2], label=1.0))
        # adding negative examples
        not_in_relevant = set(all_arg_ids) - set(query.relevant_candidates)
        for arg_id in sample(sorted(not_in_relevant), (proportion_negatives * len(query.relevant_candidates))):
            sent2 = corpus[corpus.argument_id == arg_id].argument.values[0]
            examples_list.append(InputExample(texts=[sent1, sent2], label=0.0))
        return examples_list
    
    all_arg_ids = corpus.argument_id.unique() # for later filtering
    examples_list = []
    if show_progress_bar:
        for _, query in tqdm(queries.iterrows(), total=len(queries)):
            examples_list = process_query(query, corpus, examples_list, translation_dict, all_arg_ids)
    else:
        for _, query in queries.iterrows():
            examples_list = process_query(query, corpus, examples_list, translation_dict, all_arg_ids)
    
    return examples_list