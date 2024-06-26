# Repository for GESIS-DSM at The Perspective Argument Retrieval Shared Task 2024:
This repository contains the code for the [perspective argument retrieval shared task](https://blubberli.github.io/perspective-argument-retrieval.github.io/index.html) contribution from the GESIS data science methods team.
Find the associated system description paper here [NOTE: publication will be linked as soon as it is available].

The system employs a three-step pipeline to retrieve relevant arguments given a query text and a queried socio-cultural attribute.

## Repository structure & reproduction
Find implemented functionality of SBERT-based and tf-idf-based rankers in [`src/models/rankers.py`](src/models/rankers.py).\
Find a script for generating arguments using quantized Mistral-type models in [`src/models/prompt_llm.py`](src/models/prompt_llm.py).

[`generates_arguments/`](generated_arguments/) contains generated arguments per query for the perspective scenarios for each of the shared tasks' three test sets.


[`src/notebooks/`](src/notebooks/) contains the following notebooks to reproduce our approach, results and analysis:
- [`final_submission`](src/notebooks/final_submission.ipynb): Notebook for our final submitted system.
- [`classifier_relevance`](src/notebooks/classifier_relevance.ipynb): RF classifier training.
- [`generated_test`](src/notebooks/generated_test.ipynb): Results for re-ranking with LLM-generated arguments.
- [`clustering`](src/notebooks/clustering.ipynb): Clustering analysis.
- [`style_OLS_analysis`](src/notebooks/style_OLS_analysis.ipynb): OLS analysis of stylistic features.
- [`overview`](src/notebooks/overview.ipynb): Data exploration.

## Citation
NOTE: will be added upon publication
