import re
import math
import random
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict
import logging
import json
from pathlib import Path
from nltk import word_tokenize

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from utils.my_pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from utils.my_pretrained_transformer_indexer import PretrainedTransformerIndexer

from utils.sparql_executer import get_notable_type
from utils.logic_form_util import reverse_properties, get_sub_programs, fill_sub_programs
from utils.semparse_util import lisp_to_nested_expression
from new_model.bottom_up_parser import Program

logger = logging.getLogger(__name__)
path = str(Path(__file__).parent.absolute())


# todo: There might be a discrepency between training and inference for literal representations in fill_sub_programs


@DatasetReader.register("bottom_up")
class BUParser_DatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``ComposedSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens : ``TextField`` and
        target_tokens : ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    # Parameters

    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``SpacyTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_add_end_token : bool, (optional, default=True)
        Whether or not to add `END_SYMBOL` to the end of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """

    def __init__(
            self,
            # source_tokenizer: Tokenizer = None,
            # source_token_indexers: Dict[str, TokenIndexer] = None,
            # lazy: bool = False,
            dataset: str = 'grail',
            training: bool = True,
            perfect_entity_linking: bool = True,
            delexicalization: bool = False,
            EOS: str = '[SEP]',
            eval: bool = False,
            infer: bool = False,
            decoding_steps=5,
            # 0: domain-based bc training; 1: similarity-based bc training; 2: bottom-up teacher forcing  (0 and 1 are useless now, so I have removed them)
            training_option: int = 2,
            num_data=None,
            LLM=False,
            reverse_order=False,
            shuffle=False
    ) -> None:
        super().__init__()
        # self._source_tokenizer = source_tokenizer or (lambda x: x.split())
        # In my final experiments, both source_tokenizer and source_token_indexers are not used here at all
        # self._source_tokenizer = source_tokenizer or PretrainedTransformerTokenizer("bert-base-uncased", True)
        # self._source_token_indexers = source_token_indexers or PretrainedTransformerIndexer("bert-base-uncased", True)
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self._training = training
        self._dataset = dataset
        self._perfect_el = perfect_entity_linking
        self._delexicalization = delexicalization
        self._EOS = EOS
        self._eval = eval
        self._infer = infer
        self._decoding_steps = 10  # doesn't matter if it is large
        self._training_option = training_option

        self._num_data = num_data
        self._LLM = LLM
        self._reverse = reverse_order
        self._shuffle = shuffle

        if not self._perfect_el:
            if self._dataset == 'grail':
                # with open(path + "/../el_results/grail_combined.json") as f:
                with open(path + "/../el_results/grail_combined_tiara.json") as f:
                    self._el_results = json.load(f)
            elif self._dataset == 'gq1':
                with open(path + "/../el_results/graphq_test.json") as f:
                    self._el_results = json.load(f)
            elif self._dataset == 'webq':
                # with open(path + "/../el_results/webqsp_test.json") as f:
                with open(path + "/../el_results/webqsp_test_elq.json") as f:
                    self._el_results = json.load(f)

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".json"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        if self._infer and self._dataset in ["grail", "gq1"]:  # only used for inference (also for validation)
            self._answer_types = defaultdict(lambda: [])
            if self._dataset == "grail":
                at_fn = "answer_types_grail_combined.txt"
            else:
                at_fn = "answer_types_gq1.txt"
            with open(path + f"/../answer_typing/{at_fn}", 'r') as f:
                for line in f:
                    line = line.replace("\n", '')
                    fields = line.split('\t')
                    for item in fields[1:]:
                        self._answer_types[fields[0]].append(item)

        count = 0
        with open(cached_path(file_path), 'r') as data_file:
            file_contents = json.load(data_file)
            if self._shuffle:
                random.shuffle(file_contents)
            if self._reverse:
                file_contents = file_contents[::-1]
            for item in file_contents:
                if count == self._num_data:
                    break

                if item['qid'] in [2102902009000]:  # will exceed maximum length constraint
                    continue

                if self._dataset == 'webq' and 's_expression' in item and item['s_expression'] is None \
                        and not self._infer:
                    continue

                entity_name_map = {}
                if self._perfect_el:
                    for node in item["graph_query"]["nodes"]:
                        if node["node_type"] == "entity" or node["node_type"] == "literal":
                            if node['function'] not in ['argmax', 'argmin']:
                                entity_name_map[node['id']] = node['friendly_name'].lower()

                        if self._dataset == 'webq' and "time_constraint" in node and node[
                            "time_constraint"] != "none" and node["time_constraint"][0] != "NOW":
                            entity_name_map[str(node["time_constraint"][0])] = str(node["time_constraint"][0])
                else:
                    el_results_item = self._el_results[str(item['qid'])]
                    for m in el_results_item:
                        for mid in el_results_item[m]:
                            entity_name_map[str(mid)] = m.lower()


                assert self._infer or self._training_option == 2
                if "level" in item:
                    level = item["level"]
                else:
                    level = None
                if "s_expression" in item:
                    if "graph_query" in item:
                        for node in item["graph_query"]["nodes"]:
                            if node["node_type"] == 'class':
                                gold_answer_type = node['id']
                                break
                    yield self.text_to_instance(item['question'], entity_name=entity_name_map,
                                                gold_answer_type=gold_answer_type,
                                                gold_program=item["s_expression"], level=level, qid=item['qid'])
                else:
                    yield self.text_to_instance(item['question'], entity_name=entity_name_map, qid=item['qid'])

                count += 1

    @overrides
    def text_to_instance(self,
                         question,
                         program=None,
                         label=None,
                         entity_name=None,
                         gold_program=None,
                         gold_answer_type=None,
                         level=None,
                         qid=None
                         ) -> Instance:  # type: ignore
        if not self._LLM:
            question = ' '.join(word_tokenize(question))
            question = question.replace('``', '"').replace("''", '"')
        source_string = question.lower()

        try:
            answer_types = self._answer_types[str(qid)]
        except Exception:
            answer_types = None

        qid = MetadataField(qid)

        assert self._infer or self._training_option == 2
        input_field = MetadataField(None)
        label_field = MetadataField(None)

        if gold_program is not None and self._perfect_el:  # for training, we must do perfect_el!
            sub_formulas, level_mapping = get_sub_programs(gold_program)
            filled_programs = fill_sub_programs(sub_formulas, entity_name)
            filled_programs_raw = fill_sub_programs(sub_formulas, entity_name, use_mid=True)
            # print(filled_programs)
            height = len(level_mapping)
            # gold_programs = [[] for _ in range(height)]
            gold_programs = [[] for _ in range(self._decoding_steps)]
            for k in level_mapping:
                for pid in level_mapping[k]:
                    # gold_programs[k].append(filled_programs[pid])
                    source = []
                    for eid in entity_name:
                        if eid in filled_programs_raw[pid]:
                            source.append(eid)
                    if len(source) == 0:
                        # print('wtf', gold_program_i, entity_name)
                        source = lisp_to_nested_expression(filled_programs_raw[pid])[1]  # a class name
                    if len(source) == 1:
                        source = source[0]
                    else:
                        source = set(source)
                    gold_programs[k].append(Program(source=source,
                                                    code=filled_programs[pid],
                                                    code_raw=filled_programs_raw[pid],
                                                    height=k))
            if height < self._decoding_steps:
                # fill the decoding steps with the finalized gold program
                for i in range(self._decoding_steps - height):
                    gold_programs[height + i].extend(gold_programs[height - 1])
        else:
            height, gold_programs = None, None

        # print("gold programs:", gold_programs)

        instance_dict = {"input_pair": input_field,
                         "label": label_field,
                         "question": MetadataField(question),
                         # "raw_question": MetadataField(raw_question),
                         "entity_name": MetadataField(entity_name),
                         "gold_program": MetadataField(gold_program),
                         "gold_programs": MetadataField(gold_programs),
                         "level": MetadataField(level),
                         "gold_height": MetadataField(height),
                         "gold_answer_type": MetadataField(gold_answer_type),
                         "ids": qid,
                         "answer_types": MetadataField(answer_types)}

        return Instance(instance_dict)



if __name__ == '__main__':
    reader = BUParser_DatasetReader()
    reader._read(path + '/../data/debug_grail.json')
