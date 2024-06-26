def basic_prompt(queries, profiles, stance, language="German"):
    '''Prompt template function.
    
    Args:
        queries: list of queries
        profiles: list of dict with sociocultural information per query
        stance: "pro" or "against"
        language: Language of the argument's author

    Returns:
        prompts: list of prompts
    '''
    
    prompts = []
    for (query, profile) in zip(queries, profiles):
        type_demo = list(profile.keys())[0]
        token_demo = list(profile.values())[0]

        prompt = f"Given the question {query}, generate a {stance} " + \
        f"argument a {language} speaking from a person whose {type_demo} is {token_demo}."

        prompts.append(prompt)
    return prompts


def prompt_knowledge_politics(queries, profiles, stance, language="German"):
    '''Prompt template function.
    
    Args:
        queries: list of queries
        profiles: list of dict with sociocultural information per query
        stance: "pro" or "against"
        language: Language of the argument's author

    Returns:
        prompts: list of prompts
    '''
    
    prompts = []
    for (query, profile) in zip(queries, profiles):
        type_demo = list(profile.keys())[0]
        token_demo = list(profile.values())[0]

        prompt = f"Given the question {query}, \
            use your knowledge of the Swiss political landscape to provide a {stance} argument \
             a {language} speaking person whose {type_demo} is {token_demo} would produce."
        
        prompts.append(prompt)
    return prompts

def prompt_culture(queries, profiles, stance, language="German"):
    '''Prompt template function.
    
    Args:
        queries: list of queries
        profiles: list of dict with sociocultural information per query
        stance: "pro" or "against"
        language: Language of the argument's author

    Returns:
        prompts: list of prompts
    '''
    
    prompts = []
    for (query, profile) in zip(queries, profiles):
        type_demo = list(profile.keys())[0]
        token_demo = list(profile.values())[0]

        prompt = f"Given the question {query}, \
            generate a {stance} argument a person from the {language}-speaking population \
            in Switzerland whose {type_demo} is {token_demo} would produce."

        prompts.append(prompt)
    return prompts

