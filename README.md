:mega: :mega:**(This is mainly for reproducing the results in our ACL'2023 paper. We will release the generic Pangu library in [OSU-NLP-Group/Pangu](https://github.com/OSU-NLP-Group/Pangu).)**

# Don't Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/dki-lab/Pangu/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
[![paper](https://img.shields.io/badge/Paper-ACL2023-lightgrey?style=flat-square)](https://arxiv.org/abs/2212.09736)
[![award](https://img.shields.io/badge/Award-Outstanding%20Paper%20Award-gold?style=flat-square)](https://arxiv.org/abs/2212.09736)

>A key missing capacity of current language models (LMs) is grounding to real-world environments. Most existing work for grounded language understanding uses LMs to directly generate plans that can be executed in the environment to achieve the desired effects. It thereby casts the burden of ensuring grammaticality, faithfulness, and controllability all on the LMs. We propose Pangu, a generic framework for grounded language understanding that capitalizes on the discriminative ability of LMs instead of their generative ability. Pangu consists of a symbolic agent and a neural LM working in a concerted fashion: The agent explores the environment to incrementally construct valid plans, and the LM evaluates the plausibility of the candidate plans to guide the search process. A case study on the challenging problem of knowledge base question answering (KBQA), which features a massive environment, demonstrates the remarkable effectiveness and flexibility of Pangu: A BERT-base LM is sufficient for setting a new record on standard KBQA datasets, and larger LMs further bring substantial gains. Pangu also enables, for the first time, effective few-shot in-context learning for KBQA with large LMs such as Codex.


<img width="450" alt="image" src="https://github.com/dki-lab/Pangu/assets/15921425/0945d7bf-e200-4bce-867f-550bb341e16f">
<img width="320" alt="image" src="https://github.com/dki-lab/Pangu/assets/15921425/3ec0525a-4175-4944-8980-2e3d4e1b010a">

## Walk Through Pangu with KBQA
We instantiate Pangu on knowledge base question answering (KBQA), which is representative testbed for grounded language understanding with a highly complex and heterogeneous environment.
![pangu_compressed](https://github.com/dki-lab/Pangu/assets/15921425/afb980ac-0d7a-48b2-a1ff-f39e60f20437)

## File Structure
```
pangu/
├─  acl_configs/: configuration files for training and inference
├─  data/: KBQA data files (e.g., GrailQA)
├─  ontology/: Processed Freebase ontology files
├─  answer_typing/: Answer typing results
├─  el_results/: Entity linking results 
├─  utils/:
│    ├─  bert_interface.py: Interface to BERT 
│    ├─  huggingface_interface.py: Interface to Huggingface models 
│    ├─  logic_form_util.py: Tools related to logical forms, including the exact match checker for two logical forms
│    ├─  sparql_executor.py: Sparql-related tools
│    ├─  kb_environment.py: Core functions for KB querying and constrained decoding
│    └── sparql_cache.py: Cache executions of different types of Sparql queries
├─  new_model/:
│    ├─  bottom_up_parser.py: Pangu model class
│    └── bottom_up_parser_reader.py: Pangu dataset reader class
├─  new_model/: prediction results in json
├─  run.py: Main function
└── environment.yml: yml file for conda environment 
```

## Results
### Overall Results
<img width="782" alt="image" src="https://github.com/dki-lab/Pangu/assets/15921425/04874076-5117-4979-b0f3-02fbd113a64c">

### Sample Efficiency
<img width="575" alt="image" src="https://github.com/dki-lab/Pangu/assets/15921425/6c9f244c-8252-4087-a062-20dc2f8f450a">

### Strong Generalizability
<img width="627" alt="image" src="https://github.com/dki-lab/Pangu/assets/15921425/0dbc08bc-2d29-41cd-9106-9e82c555ae5f">

## Reproducing Our Results
### Environment Setup
Please configure your own conda environment using [environment.yml](https://github.com/dki-lab/Pangu/blob/main/environment.yml). Replace `[your_conda_path]` in that file with the path of your local anaconda folder, and then create the environment with `conda env create -f environment.yml`.

### Training & Inference
```
PYTHONHASHSEED=23 python run.py
    \ train
    \ acl_configs/grail_train.jsonnet
    \ --include-package
    \ new_model.bottom_up_parser
    \ --include-package
    \ new_model.bottom_up_parser_reader
    \ --include-package
    \ utils.huggingface_interface
    \ -s
    \ [output_dir]
```

To do inference with a saved model, use the first configuration in `launch.json`, or do
```
PYTHONHASHSEED=23 python run.py
    \ predict
    \ [output_dir]/model.tar.gz
    \ [path_to_file] (e.g., grail_v1.0_test_public.json)
    \ --include-package
    \ new_model.bottom_up_parser
    \ --include-package
    \ new_model.bottom_up_parser_reader
    \ --include-package
    \ utils.huggingface_interface
    \ -output-file
    \ predictions.txt
    \ --use-dataset-reader
    \ -o
    \ "{'model': {'infer': true}, 'validation_dataset_reader': {'infer': true, 'perfect_entity_linking': false}}"
```

In `utils.sparql_executer.py`, replace "http://127.0.0.1:3094/sparql" with your own SPARQL endpoint.

Configure your experiments following configuration files under `acl_configs`.

### Experiments with LLMs
Our original experiments with LLMs were done with Codex. However, since March 2023, Codex has been deprecated by OpenAI. We will adjust and upload this part of code soon.
