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
├─  trained_models.md: the links to our trained models; download them to make predictions
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
PYTHONHASHSEED=23 python run.py \
    train \
    acl_configs/grail_train.jsonnet \
    --include-package \
    new_model.bottom_up_parser \
    --include-package \
    new_model.bottom_up_parser_reader \
    --include-package \
    utils.huggingface_interface \
    -s \
    [output_dir]
```
To train the model with multiple cards using DDP, uncomment the `distributed` field in the config file.
Note that, training can be quite slow at an earlier stage, but it will be faster when more SPARQL queries are executed and cached.

To do inference with a saved model, use the first configuration in `launch.json`, or do
```
PYTHONHASHSEED=23 python run.py \
    predict \
    [output_dir]/model.tar.gz \
    [path_to_file] \
    --include-package \
    new_model.bottom_up_parser \
    --include-package \
    new_model.bottom_up_parser_reader \
    --include-package \
    utils.huggingface_interface \
    --output-file \
    predictions.txt \
    --use-dataset-reader \
    --cuda 0 \
    -o \
    "{'model': {'infer': true}, 'validation_dataset_reader': {'infer': true, 'perfect_entity_linking': false}}"
```

To do inference with an OpenAI API, do
```
PYTHONHASHSEED=23 python run.py \
  predict \
  openai_eval/allennlp_dummy/model.tar.gz \
  data/debug_grail.json \
  --include-package \
  openai_eval.bottom_up \
  --include-package \
  new_model.bottom_up_parser_reader \
  --include-package \
  utils.huggingface_interface \
  --output-file \
  predictions.txt \
  --use-dataset-reader \
  -o \
 "{'validation_dataset_reader': {'infer': true, 'perfect_entity_linking': false}}"
```
Here `openai_eval/allennlp_dummy/model.tar.gz` is only a dummy model that does not really matter. The only reason to have it is that allennnlp predict requires to first load load some model weights first, while we only use the API here and do not have any real weights to load. In `openai_eval/bottom_up.py`, put your openai_key in [line 155](https://github.com/dki-lab/Pangu/blob/ad8431bce41e30cc66dae00d328fa08607e4cd07/openai_eval/bottom_up.py#L155). To configure the retrieval pool for in-context learning, please see [these lines](https://github.com/dki-lab/Pangu/blob/ad8431bce41e30cc66dae00d328fa08607e4cd07/openai_eval/bottom_up.py#L201-L225).

In `utils.sparql_executer.py`, replace "http://127.0.0.1:3094/sparql" with your own SPARQL endpoint.

Configure your experiments following configuration files under `acl_configs`.

### Experiments with LLMs
Our original experiments with LLMs were done with Codex. However, since March 2023, Codex has been deprecated by OpenAI. We will adjust and upload this part of code soon.

### Training Time
We've noted concerns about the prolonged training times on GrailQA, and would like to address some frequently asked questions:

**Q1:** The projected training time for one epoch on GrailQA exceeds 100 hours, leading to a total estimated duration of over 20 days. Is this accurate? <br>
**A1:** During the initial phase of training, the primary hindrance to speed is the execution of SPARQL queries. As the results of these queries get cached over time, the training pace significantly accelerates. For context, when training on GrailQA with four A6000 cards, an epoch typically completes in about 15 hours. The total estimated duration you're seeing is a product of the estimated per-epoch training time and the maximum number of epochs specified in your configuration file. However, in practice, we usually don't train for the maximum number of epochs. Implementing early stopping (patience=3), the model generally completes training in just 4 or 5 epochs. This suggests convergence occurring by the first or second epoch.

**Q2:** Why does validation seem sluggish? <br>
**A2:** The delay in validation is largely due to the caching of SPARQL executions. The process will speed up as more queries get cached. For an enhanced validation process, I recommend you uncomment every `if True` line in [kb_environment.py](https://github.com/dki-lab/Pangu/blob/main/utils/kb_environment.py) and comment out the `if self.training` lines preceding them. I recognize this isn't the most intuitive code design, and I aim to refine its structure for clearer functionality and better user experience in the near future.

**Q3:** How to do distributed training? <br>
**A3:** AllenNLP natively supports DDP (Distributed Data Parallel) training. To activate this feature, just uncomment the 'distributed' argument in your configuration file.

Additionally, we believe the cache file can be very useful for your experiments, so we also upload the cached SPARQL queries over GrailQA to OneDrive for your convenience: [execution.json](https://1drv.ms/u/s!AuJiG47gLqTzoX3hugkhIfIAOqoO?e=kqtTiy).


## Acknowledgements
The authors would like to thank Percy Liang, Jiawei Han, Jonathan Berant, Huan Sun, and other colleagues from the OSU NLP group for their valu- able feedback. The authors would also like to thank Shijie Chen and Chan Hee Song for proof-of-concept implementation of Pangu on other tasks, Yiheng Shu for sharing their entity linking results, and Tianbao Xie for clarifications on UnifiedSKG. This research was supported in part by `ARL W911NF2220144`, `NSF OAC 2112606`, and Ohio Supercomputer Center.

### Citation
If you find our work helpful, please cite our ACL paper as follows.
```
@inproceedings{gu-etal-2023-dont,
    title = "Don{'}t Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments",
    author = "Gu, Yu  and
      Deng, Xiang  and
      Su, Yu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.270",
    pages = "4928--4949",
    abstract = "A key missing capacity of current language models (LMs) is grounding to real-world environments. Most existing work for grounded language understanding uses LMs to directly generate plans that can be executed in the environment to achieve the desired effects. It thereby casts the burden of ensuring grammaticality, faithfulness, and controllability all on the LMs. We propose Pangu, a generic framework for grounded language understanding that capitalizes on the discriminative ability of LMs instead of their generative ability. Pangu consists of a symbolic agent and a neural LM working in a concerted fashion: The agent explores the environment to incrementally construct valid plans, and the LM evaluates the plausibility of the candidate plans to guide the search process. A case study on the challenging problem of knowledge base question answering (KBQA), which features a massive environment, demonstrates the remarkable effectiveness and flexibility of Pangu: A BERT-base LM is sufficient for setting a new record on standard KBQA datasets, and larger LMs further bring substantial gains.Pangu also enables, for the first time, effective few-shot in-context learning for KBQA with large LMs such as Codex.",
}
```
