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
