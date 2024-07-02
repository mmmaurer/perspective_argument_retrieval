# Repository for GESIS-DSM at The Perspective Argument Retrieval Shared Task 2024:
This repository contains the code for the [perspective argument retrieval shared task](https://blubberli.github.io/perspective-argument-retrieval.github.io/index.html) contribution from the GESIS data science methods team.
Find the associated system description paper here [NOTE: publication will be linked as soon as it is available].

The system employs a three-step pipeline to retrieve relevant arguments given a query text and a queried socio-cultural attribute.

## Data for reproducing this work
To reproduce our work, download the respective dataset cycle from [this repository](https://blubberli.github.io/perspective-argument-retrieval.github.io/index.html).
Per dataset cycle, it should have the following structure:
- `dataset_cycle_n/`
    - `baseline_queries/`
        - `train.jsonl`
        - `val.jsonl`
        - `test.jsonl`
    - `perspective_queries/`
        - `train.jsonl`
        - `val.jsonl`
        - `test.jsonl`
    - `corpus.jsonl`

Given this structure, the path `dataset_cycle_n/` can then be used as data path in our implementation to load the corpus and queries from.

## Repository structure & reproduction
Find implemented functionality of SBERT-based and tf-idf-based rankers in [`src/models/rankers.py`](src/models/rankers.py).\
Find a script for generating arguments using quantized Mistral-type models in [`src/models/prompt_llm.py`](src/models/prompt_llm.py).

[`generates_arguments/`](generated_arguments/) contains generated arguments per query for the perspective scenarios for each of the shared tasks' three test sets.


[`src/notebooks/`](src/notebooks/) contains the following notebooks to reproduce our approach, results and analysis:
- [`final_submission`](src/notebooks/final_submission.ipynb): Notebook for our final submitted system.
- [`classifier_relevance`](src/notebooks/classifier_relevance.ipynb): RF classifier training.
- [`generated_test`](src/notebooks/generated_test.ipynb): Results for re-ranking with LLM-generated arguments.
- [`clustering`](src/notebooks/clustering.ipynb): Clustering analysis.
- [`logistic_regression.Rmd`](src/notebooks/logistic_regression.Rmd): R-Markdown notebook for the regression analysis of stylistic features. Find the results [here](src/notebooks/logistic_regression.html).
- [`lm-prep-cycle1`](src/notebooks/lm-prep-cycle1.ipynb): Preparation for the regression analysis of stylistic features.
- [`overview`](src/notebooks/overview.ipynb): Data exploration.

[`data/corpus_de_lm_1_nodummy_nomv.csv`](data/corpus_de_lm_1_nodummy_nomv.csv): Data for reproducing the OLS regression analysis.

## Citation
NOTE: will be added upon publication
