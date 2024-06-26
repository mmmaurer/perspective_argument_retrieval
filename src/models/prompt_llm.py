import argparse

import pandas as pd
import torch
from transformers import MistralForCausalLM, AutoTokenizer, QuantoConfig

from data_input.load_data import load_corpus, load_queries
from prompt_templates import prompt_culture

def generate_arguments(model, tokenizer, prompts, batch_size, max_new_tokens):
    '''Generates arguments for a given list of prompts.
    
    Args:
        model: MistralForCausalLM model
        tokenizer: AutoTokenizer
        prompts: list of prompts
        batch_size: batch size for inference
        max_new_tokens: maximum number of tokens to generate
    
    Returns:
        list of arguments
    '''
    arguments = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True). \
            to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        del inputs  # Free up memory
        arguments.extend(tokenizer.batch_decode(outputs,
                                                skip_special_tokens=True))
    return arguments

def postprocess_arguments(queries, arguments):
    '''Removes queries from arguments and returns a
    pandas DataFrame with columns "query" and "argument"
    
    Args:
        queries: list of queries
        arguments: list of arguments
    
    Returns:
        pandas DataFrame with columns "query" and "argument"
    '''
    # Removing queries and newlines, <|im_end|> tokens
    processed_arguments = [argument.replace(queries[i], ""). \
                           replace("\n", ""). \
                            replace("<|im_end|>", "") 
                            for i, argument in enumerate(arguments)]
    return pd.DataFrame({"query": queries, "argument": processed_arguments})

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str,
                        default="occiglot/occiglot-7b-eu5-instruct")
    parser.add_argument("--data_path", type=str,
                        default="../../data/")
    parser.add_argument("--output_path", type=str,
                        default="../../outputs/")
    parser.add_argument("--scenario", type=str, 
                        choices=["baseline", "perspective"],
                        default="baseline")
    parser.add_argument("--split", type=str,
                        choices=["train", "dev", "test"],
                        default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--quantization", type=str, default="int8")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--language", type=str, default="German")
    parser.add_argument("--stance", type=str, default="pro")
    args = parser.parse_args()

    # Load data
    queries = load_queries(args.data_path, args.split)
    profiles = load_corpus(args.data_path, args.split, args.scenario)

    # Load model
    quantization_config = QuantoConfig(quantization=args.quantization)
    model = MistralForCausalLM.from_pretrained(args.model_name, 
                                               config=quantization_config,
                                               torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Generate prompts
    prompts = prompt_culture(queries, profiles, args.stance, args.language)

    # Generate arguments
    arguments = generate_arguments(model, tokenizer, prompts,
                                   args.batch_size, args.max_new_tokens)
    
    # Postprocess arguments
    df = postprocess_arguments(queries, arguments)

    # Save arguments
    df.to_csv(f"{args.output_path}/{args.split}_arguments.csv", index=False)

