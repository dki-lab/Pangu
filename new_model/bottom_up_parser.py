import allennlp
import time
import logging
import functools
import numpy as np

from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict

from utils import logic_form_util
from utils.logic_form_util import same_logical_form, lisp_to_sparql, postprocess_raw_code, get_derivations_from_lisp, \
    get_canonical_lisp
from utils.sparql_executer import execute_query
from utils.semparse_util import lisp_to_nested_expression, get_nesting_level

import numpy
import math
import random
import re
import json
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import Average

from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer


# todo: in termination check, may also need to consider whether the highest program is a finalized/valid one

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds for {func.__name__}")
        return value

    return wrapper_timer


def smbop_loss(logits, label):
    assert len(logits) == len(label)
    normalized_scores = allennlp.nn.util.masked_log_softmax(logits)

    return - normalized_scores * label.squeeze().sum(-1)


class Program:
    def __init__(self,
                 source: Union[Set, str] = None,
                 code: str = '',  # used for PLM classifier
                 code_raw: str = '',  # original code (i.e., code with mids)
                 function: str = None,
                 height: int = -1,
                 execution: Union[Set, str] = None,
                 finalized: bool = False,
                 dummy: bool = False,  # dummy programs that make no sense; only for training
                 derivations: Dict = None):
        """
        :param source: anchor entities/literals
        :param code: programs with readable entity names
        :param code_raw: original programs
        :param function: function name of the outmost subprogram
        :param height: height
        :param execution: execution results or an arg class
        :param finalized: whether it is a finalized program
        :param derivations: relations paths (optionally with comparators) indexed by different source nodes
        """
        self.source = source
        self.code = code
        self.code_raw = code_raw
        self.function = function
        self.height = height
        self.execution = execution
        self.finalized = finalized
        self.derivations = derivations  # (comments: I think derivations is only used for get reachable classes?)
        self.dummy = dummy

    def execute(self, kb_engine=None):
        if kb_engine is None:  # for training
            # if True:  # todo: use another flag for this
            try:
                if isinstance(self.execution, tuple):
                    self.execution = self.execution[0](*self.execution[1:])
            except IndexError:  # some string constants in WebQSP
                self.execution = set()
        else:
            # isinstance(self.execution, str) for
            if not isinstance(self.execution, set) and not isinstance(self.execution, str):
                # self.execution = self.execution[0](*self.execution[1:])
                processed_code_raw = postprocess_raw_code(self.code_raw)
                sparql_query = lisp_to_sparql(processed_code_raw)
                try:
                    # execution = execute_query(sparql_query)
                    execution = kb_engine.execute_SPARQL(sparql_query)
                    if isinstance(execution, list):
                        execution = set(execution)
                except Exception:
                    execution = set()

                self.execution = execution

    def is_cvt(self, kb_engine):
        try:
            # assert isinstance(self.execution, set) or isinstance(self.execution, list)
            types = kb_engine.get_classes_for_variables(self.execution, cvt_check=True)
            if len(types) == 0:
                return False

            cvt = True
            for t in types:
                if t not in kb_engine.cvt_types:
                    cvt = False
                    break
            return cvt
        except Exception:
            print("is_cvt error:", self.code_raw)
            return False

    def __str__(self):
        return self.code_raw


@Model.register("bottom_up")
class BottomUpParser(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            hidden_size: int = -1,
            source_embedder: TextFieldEmbedder = None,
            dropout=0.1,
            infer=False,
            beam_size=5,
            decoding_steps=4,  # 5 for grail; 4 for graph
            training_option: int = 0,
            val_option: int = None,
            loss_option: int = 0,  # 0 for bce; 1 for softmax + (b)ce
            device: int = -1,  # device identifier for writing to the cache
            EOS: str = '[SEP]',
            encoder_decoder: str = None,  # whether use pre-trained encoder-decoders (e.g., T5) for scoring
            em_augmentation: bool = False,  # whether consider programs equivalent to gold programs for training
            dataset='grail'
    ) -> None:
        super().__init__(vocab)
        # Dense embedding of source vocab tokens.
        self._enc_dec = encoder_decoder
        self._EOS = EOS
        if self._enc_dec is None:
            self._source_embedder = source_embedder
            self._dropout = torch.nn.Dropout(p=dropout)
            self.cls_layer = nn.Linear(hidden_size, 1)

            model_name = self._source_embedder.token_embedder_tokens.model_name

            if self._EOS == '[SEP]':
                self._source_tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif self._EOS == '</s>':
                self._source_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            if "t5" in self._enc_dec:
                self._tokenizer = T5Tokenizer.from_pretrained(self._enc_dec)
                self._enc_dec_model = T5ForConditionalGeneration.from_pretrained(self._enc_dec)
                if self._enc_dec not in ["t5-small", "t5-base"]:
                    self._enc_dec_model.decoder.gradient_checkpointing = True
                    self._enc_dec_model.encoder.gradient_checkpointing = True
            else:
                pass  # not supported yet

        if loss_option == 0:  # todo: this is wrong; remove later
            self._loss = nn.BCEWithLogitsLoss()
        elif loss_option == 1:
            self._loss = nn.CrossEntropyLoss()  # only for pytorch > 1.10

        self._device_id = device

        self._infer = infer
        self._beam_size = beam_size
        self._decoding_steps = decoding_steps
        self._training_option = training_option
        self._em_augmentation = em_augmentation
        if val_option is None:
            self._val_option = training_option
        else:
            self._val_option = val_option

        if self._infer or training_option == 2 or val_option == 2:
            from utils.kb_environment import Computer
            self._computer = Computer(dataset=dataset)
        else:
            self._computer = None

        self._dataset = dataset

        if not self._infer and training_option != 2 and val_option != 2:
            self._precision = Average()
            self._recall = Average()
            self._accuracy = Average()
            self._F1 = Average()
        elif val_option == 2 and training_option != 2:
            self._precision = Average()
            self._recall = Average()
            self._accuracy = Average()
            self._exact_match = Average()
            self._exact_match_iid = Average()
            self._exact_match_comp = Average()
            self._exact_match_zero = Average()
            self._F1 = Average()
        else:
            self._exact_match = Average()
            self._exact_match_iid = Average()
            self._exact_match_comp = Average()
            self._exact_match_zero = Average()
            self._F1 = Average()

        self._init_epoch = 0  # used to track the change of epochs

        # This was used to get the score distribution for seen and unseen logical forms during inference
        # self._lf_pool = set()
        # with open("data/grailqa_v1.0_train.json") as f:
        #     data = json.load(f)
        #     for item in data:
        #         self._lf_pool.add(get_canonical_lisp(item["s_expression"]))
        #
        # print(len(self._lf_pool))

    # @timer
    @overrides
    def forward(
            self,  # type: ignore
            input_pair: Dict[str, torch.LongTensor],
            label: torch.LongTensor,
            question: List[str],
            # for entity: key->mid, value->friendly name; for value: key->value w type, value->value w/o type
            entity_name: List[Dict],
            # gold programs; used for EM evaluation
            gold_program: List[str],
            # gold programs for each decoding step, entities are replaced with surface forms; used for training
            gold_programs: List[List[List[str]]],
            level: List[str],
            gold_height: List[int],
            gold_answer_type=None,
            ids=None,
            answer_types=None
    ) -> Dict[str, torch.Tensor]:
        if self._enc_dec is None:
            self._device = next(self._source_embedder.parameters()).device
        else:
            self._device = next(self._enc_dec_model.parameters()).device

        # cache the sparql executions to file every epoch (this is clumsy, but I don't want to modify the framework
        # code of allennlp)
        if not self._infer and self.epoch != self._init_epoch:
            self._init_epoch = self.epoch
            # this is an expedient way for thread safe writing with multi cards
            if self._device.index == self._device_id:
                self._computer._cache.cache_results()
        elif self._infer and str(ids[0]) in ["test-13230", "2100263012000", "WebQTest-2031"]:
            # cache the results during prediction
            self._computer._cache.cache_results()

        # logging.info(f"Process question: {question}")
        if self._computer is not None:
            self._computer.set_training(training=self.training)

        # entity_name[0]["750^^http://www.w3.org/2001/XMLSchema#integer"] = '750'
        predictions = None
        if not self._infer and gold_programs[0] is None:  # training_option != 2
            pooler_output = self._source_embedder(input_pair)
            logits = self.cls_layer(self._dropout(pooler_output))

            output_dict = {"logits": logits}
            label.unsqueeze_(dim=-1)

            output_dict["loss"] = self._loss(logits, label.float())

            predictions = (F.sigmoid(logits) > 0.5).type_as(label)

            #  metrics calculation
            self._accuracy(float((predictions == label).sum()) / float(label.shape[0]))
            tp = float(((predictions == label) & (predictions == 1)).sum())
            if float((predictions == 1).sum()) > 0:
                precision = tp / float((predictions == 1).sum())
            else:
                precision = 0
            self._precision(precision)
            if float((label == 1).sum()) > 0:
                recall = tp / float((label == 1).sum())
            else:
                recall = 1
            self._recall(recall)
            if recall > 0 and precision > 0:
                self._F1(2 / (1 / precision + 1 / recall))
            else:
                self._F1(0)

            return output_dict
        else:
            if gold_programs[0] is not None:
                # re-organize gold_programs: from batch_size * heights to heights * batch_size
                gold_programs = np.array(gold_programs, dtype=object)
                if len(gold_programs.shape) == 3:
                    gold_programs = np.squeeze(gold_programs, axis=2)
                gold_programs = np.transpose(gold_programs, [1, 0])
                gold_programs = list(map(lambda x: list(x), gold_programs))

            # The finalized programs will have length decoding steps; each item has length batch size
            # todo: for now only works for batch size 1
            programs: List[List[List[Program]]] = []
            programs_indexed = [defaultdict(lambda: []) for _ in range(len(question))]  # len batch_size
            assert len(programs_indexed) == 1
            highest_scores = [-1e32 for _ in
                              range(len(question))]  # used to track the highest score for termination check
            # todo: may also track highest finalized programs
            finalized_programs = [[] for _ in range(len(question))]
            loss_each_step = []
            beam_logits = None
            # for decoding_step in range(self._decoding_steps):
            num_candidates = 0
            if self.training and self._training_option == 2:
                # decoding_steps = self._decoding_steps
                # decoding_steps = min(int(gold_height[0]) + 1, self._decoding_steps)
                decoding_steps = int(gold_height[0]) + 1
                # decoding_steps = min(int(gold_height[0]), self._decoding_steps)
            elif self._infer or self._val_option == 2:  # for prediction and validation
                decoding_steps = self._decoding_steps
                # decoding_steps = int(gold_height[0]) + 1
            for decoding_step in range(decoding_steps):  # +1 to get negative samples with larger heights
                candidate_programs = []
                # if decoding_step == decoding_steps - 1:
                #     print("for debugging")
                if decoding_step == 0:
                    for i, en in enumerate(entity_name):
                        ini_programs_i = self._computer.get_initial_programs(en, answer_types[i], gold_answer_type[i])
                        candidate_programs.append(ini_programs_i)
                else:
                    for i in range(len(programs_indexed)):  # for i in range(batch_size)
                        candidate_programs_i = self._computer.get_admissible_programs(programs[decoding_step - 1][i],
                                                                                      programs_indexed[i],
                                                                                      entity_name[i]
                                                                                      )
                        candidate_programs.append(candidate_programs_i)
                        num_candidates += len(candidate_programs_i)

                if self.training and len(candidate_programs[0]) == 0:
                    # append gold programs for training
                    candidate_programs_i = []
                    for program in gold_programs[decoding_step]:
                        if isinstance(program, list):
                            for p in program:
                                wrapped_program = self._wrap_program(p, entity_name[0])
                                if wrapped_program is not None:
                                    candidate_programs_i.append(wrapped_program)
                        else:
                            wrapped_program = self._wrap_program(program, entity_name[0])
                            if wrapped_program is not None:
                                candidate_programs_i.append(wrapped_program)

                    candidate_programs[0] = candidate_programs_i

                if len(candidate_programs[0]) == 0:  # normally due to all beam programs being finalized
                    break
                else:
                    # elif len(candidate_programs[0]) > 1 or self.training:
                    # print(len(candidate_programs[0]))
                    # During training, beam_logits may not be strictly in order for two reasons:
                    # 1) Gold ids are manually populated into the beam
                    # 2) The scores for beam items are recomputed for backprop with dropout
                    new_beam_programs, scores, beam_logits, labels, loss_logits, gold_candidates = self._get_top_candidates(
                        candidate_programs, question,
                        decoding_step,
                        gold_programs,
                        entity_name)

                    if self.training:
                        # todo: Currently, I am not normalizing the labels for CrossEntropy
                        # labels is only used for loss calculation
                        # step_loss = self._loss(beam_logits, labels)
                        step_loss = self._loss(loss_logits, labels)
                        loss_each_step.append(step_loss)

                        if decoding_step != decoding_steps - 1:  # not the last step
                            for bp in new_beam_programs[0]:
                                bp.execute()

                    else:
                        assert len(scores) == len(highest_scores)
                        termination_flag = False
                        for i in range(len(highest_scores)):
                            # todo: for batching, need to handle the asynchronous termination issue
                            if len(scores[i]) > 0 and scores[i][0] > highest_scores[0]:
                                # for inference, scores should be in descending order
                                highest_scores[0] = scores[i][0]
                            elif decoding_step > 0:
                                termination_flag = True
                                break
                        if termination_flag:
                            break

                    # update beam_programs to the current step
                    beam_programs = new_beam_programs

                # else:  # len(beam_programs) == 1
                #     beam_programs = candidate_programs
                #     # if self.training:
                #     #     gold_candidates = candidate_programs  # be consistent with training acceleration

                programs.append([[bp for bp in beam_programs[0] if not bp.dummy]])
                for i, candidates in enumerate(beam_programs):
                    for p in candidates:
                        if p.dummy:
                            continue
                        if isinstance(p.source, set):
                            p.source = tuple(p.source)
                        programs_indexed[i][p.source].append(p)

                # if not self.training:
                #     programs.append(beam_programs)
                #     for i, candidates in enumerate(beam_programs):
                #         for p in candidates:
                #             if isinstance(p.source, set):
                #                 p.source = tuple(p.source)
                #             programs_indexed[i][p.source].append(p)
                # else:  # for training acceleration; only gold programs are expanded
                #     programs.append(gold_candidates)
                #     for i, candidates in enumerate(gold_candidates):
                #         for p in candidates:
                #             if isinstance(p.source, set):
                #                 p.source = tuple(p.source)
                #             programs_indexed[i][p.source].append(p)

            if beam_logits is not None:  # only for training; for inference, the beam programs are already ranked
                _, indices = torch.sort(beam_logits, descending=True)
                # beam_programs = torch.gather(beam_programs, dim=-1, index=indices)
                beam_programs = [[beam_programs[i][j] for j in indices[i]] for i in range(len(beam_programs))]
            # elsewise, the final beam only comprises one candidate

            try:
                if self.training:
                    predictions = beam_programs[0][0]
                else:
                    finalized = False
                    for p in beam_programs[0]:  # only works for batch size 1
                        selection = False
                        if p.finalized:
                            if p.execution is None:
                                selection = True
                            elif isinstance(p.execution, int) and p.execution != 0:
                                selection = True
                            elif not isinstance(p.execution, int) and len(p.execution) > 0:
                                if not p.is_cvt(self._computer):
                                    selection = True
                        if selection:
                            finalized = True
                            predictions = p
                            break
                    if self._dataset == 'webq' and finalized:  # because torch.topk is unstable
                        entities = []
                        for e in entity_name[0]:
                            entities.append(e)
                        if isinstance(predictions.source, str):
                            eid = entities.index(predictions.source)
                        else:
                            eid = entities.index(predictions.source[0])
                        for p in beam_programs[0]:
                            if p.code == predictions.code and p.source != predictions.source:
                                if isinstance(p.source, str):
                                    peid = entities.index(p.source)
                                else:
                                    peid = entities.index(p.source[0])
                                if peid < eid:
                                    eid = peid
                                    predictions = p

                    if not finalized:  # todo: here still need to filter null answer
                        if len(beam_programs[0]) > 0:
                            predictions = beam_programs[0][0]
                        elif len(candidate_programs_i) > 0:
                            predictions = candidate_programs_i[0]  # ideally, this should never happen
                        else:
                            predictions = Program()
                            print("Unexpected!!!!")

                if predictions.code_raw != '':
                    predictions.code_raw = postprocess_raw_code(predictions.code_raw)
                # print("\nQuestion:", question[0])
                # print("Gold:", gold_program[0])
                # print("Predict:", predictions.code_raw)
                if gold_program[0] is not None:
                    em = same_logical_form(predictions.code_raw, gold_program[0])
                else:
                    em = 0
                # print("total passes:", num_candidates)
            except UnboundLocalError:  # beam_programs referenced before assignment
                # possible reasons for this:
                # 1. empty entity linking results before implementing superlatives
                # 2. no admissible relations for entities
                em = 0
                print("question:", question)

            # try:
            #     print(float(torch.softmax(scores[0], dim=0).max().data),
            #           # level[0],
            #           get_canonical_lisp(predictions.code_raw) in self._lf_pool,
            #           file=open("scores_bert_grail_new.txt", 'a'))
            # except Exception:
            #     pass  # not sure what's wrong. I don't care

            self._exact_match(em)
            if level[0] == "i.i.d.":
                self._exact_match_iid(em)
            if level[0] == "compositional":
                self._exact_match_comp(em)
            if level[0] == "zero-shot":
                self._exact_match_zero(em)

            try:
                loss = torch.stack(loss_each_step).mean()
            except Exception:  # there is no candidate relation starting from the topic entity for training
                loss = torch.tensor(0.0).to(self._device)
                loss.requires_grad = True  # to avoid RuntimeError

            output_dict = {"loss": loss,
                           "predictions": predictions,
                           "ids": ids}

            return output_dict

    # @timer
    def _get_top_candidates(self, candidate_programs: List[List[Program]],
                            question: List[str],
                            # List[str] when all steps only have 1 gold sub-tree, or List[List[str]] otherwise
                            decoding_step: int = -1,  # only used for training
                            gold_programs=None,  # gold sub-programs of each step
                            entity_name=None
                            ):
        # len(top_candidates) = batch_size; len(top_candidates[0]) = beam_size
        top_candidates: List[List[Program]] = []
        gold_candidates: List[List[Program]] = []  # this could be used to accelerate training
        scores: List[torch.FloatTensor] = []

        if self.training:
            label_list = []
            beam_scores_list = []
            logits_for_bce = []
        else:
            label_list = None
            beam_scores_list = None
            logits_for_bce = None

        for i, candidate_programs_i in enumerate(candidate_programs):  # for each batch instance
            if self.training:
                # if we do bottom-up teacher forcing, gold programs should appear in the candidate list for
                # every step; this is also handled within self._get_gold_ids (only for training)
                # todo (11/05) after not returning the same program as a admissible candidate, for step gold_height+1, we always need to do wrap_program. Optimize this
                gold_ids, em_gold_ids = self._get_gold_ids(gold_programs[decoding_step][i], candidate_programs_i,
                                                           entity_name[i])
                # downsampling for training
                # if self.training and len(candidate_programs_i) > 10:
                #     down_sampled_candidate_i = []
                #     gold_ids.sort(reverse=True)
                #     for gold_id in gold_ids:
                #         down_sampled_candidate_i.append(candidate_programs_i[gold_id])
                #         candidate_programs_i.pop(gold_id)
                #     gold_ids = [i for i in range(len(down_sampled_candidate_i))]
                #
                #     remaining = random.sample(candidate_programs_i, 10 - len(down_sampled_candidate_i))
                #     down_sampled_candidate_i.extend(remaining)
                #
                #     candidate_programs_i = down_sampled_candidate_i

            pairs = []
            for p in candidate_programs_i:
                pairs.append((question[i], p.code))

            # todo: for now there is no batch operation across different batch instances. decide it later
            with torch.no_grad():
                if self._enc_dec is None:
                    score = self._score_pairs(pairs)
                    score.squeeze_(1)
                else:
                    score = self._score_pairs_with_t5(pairs)

            assert len(candidate_programs_i) == len(score)
            # (beam_size, )
            if self.training:
                # top_scores, top_indices = torch.topk(score, k=min([len(pairs), self._beam_size]))
                # 20 is hard-coded here for now to increase the set for negative samples during training
                top_scores, top_indices = torch.topk(score, k=min([len(pairs), 15]))
            else:
                # for inference, we will need to filter some invalid programs, so we want to have a larger
                # initial beam here
                top_scores, top_indices = torch.topk(score, k=min([len(pairs), self._beam_size + 5]))

            if self.training:
                beam_indices = torch.zeros(min([len(pairs), self._beam_size]), device=top_indices.device).long()

                logits_indices = torch.zeros(len(top_indices), device=top_indices.device).long()
                label = torch.zeros(len(logits_indices), device=top_indices.device)

                for j in range(len(gold_ids)):
                    label[j] = 1  # for BCE
                    beam_indices[j] = gold_ids[j]
                    logits_indices[j] = gold_ids[j]

                remaining = len(beam_indices) - len(gold_ids)
                remaining_logits = len(logits_indices) - len(gold_ids)

                for top_id in top_indices:  # populate the remaining of the beam
                    if remaining == 0:
                        break
                    if top_id not in gold_ids:
                        beam_indices[-remaining] = top_id
                        remaining -= 1

                for top_id in top_indices:  # populate the remaining of the beam
                    if remaining_logits == 0:
                        break
                    if top_id not in gold_ids:
                        logits_indices[-remaining_logits] = top_id
                        if self._em_augmentation and top_id in em_gold_ids:
                            label[-remaining_logits] = 1
                        remaining_logits -= 1

                # separate beam scores (for beam search) and logits (for loss calculation)
                beam_scores = torch.gather(score, dim=-1, index=beam_indices)

                logits_pairs = [pairs[pid] for pid in
                                logits_indices]
                if decoding_step > 0 and gold_programs[decoding_step - 1][0] != gold_programs[decoding_step][0]:
                    # use the gold programs from the previous step as negative samples
                    if isinstance(gold_programs[decoding_step - 1][0], Program):
                        logits_pairs.append((question[i], gold_programs[decoding_step - 1][0].code))
                        label = F.pad(label, (0, 1), "constant", 0)
                    else:
                        for p in gold_programs[decoding_step - 1][0]:
                            logits_pairs.append((question[i], p.code))
                        label = F.pad(label, (0, len(gold_programs[decoding_step - 1][0])), "constant", 0)

                if self._enc_dec is None:
                    logits = self._score_pairs(logits_pairs)
                    logits.squeeze_(1)
                else:
                    logits = self._score_pairs_with_t5(logits_pairs)

                label_list.append(label)
                beam_scores_list.append(beam_scores)
                logits_for_bce.append(logits)

                # dump training pairs to file
                # for logit_pair, l in zip(logits_pairs, label):
                #     print(f"{decoding_step}\t{logit_pair[0]}\t{logit_pair[1]}\t{int(l.data)}",
                #           file=open(f"training_dump/{self._dataset}_training_{self._device.index}.txt", 'a'))

            if not self.training:
                top_candidates_i = []
                score_ids = []
                for i, idx in enumerate(top_indices):
                    if len(top_candidates_i) == self._beam_size:
                        break
                    candi = candidate_programs_i[idx]
                    candi.execute(self._computer)
                    if isinstance(candi.source, str):
                        # when candi is a finalized ARG program, candi.execution is None, which is fine
                        if isinstance(candi.execution, set):
                            if len(candi.execution) == 0 or (
                                    list(candi.execution)[0] == candi.source and len(candi.execution) == 1):
                                continue
                        else:  # COUNT function
                            if candi.execution == 0:
                                continue
                    top_candidates_i.append(candi)
                    score_ids.append(i)

                top_candidates.append(top_candidates_i)
                scores.append(top_scores[score_ids])
            else:  # only do gold program population for training
                scores.append(top_scores)
                top_candidates_i = [candidate_programs_i[idx] for idx in beam_indices]
                top_candidates.append(top_candidates_i)

                gold_candidates_i = [candidate_programs_i[idx] for idx in gold_ids]
                gold_candidates.append(gold_candidates_i)

        if self.training:
            label_list = torch.stack(label_list, dim=0)  # (>=batch_size, beam_size)
            beam_scores_list = torch.stack(beam_scores_list, dim=0)  # (batch_size, beam_size)
            logits_for_bce = torch.stack(logits_for_bce, dim=0)
            assert len(label_list) == len(logits_for_bce)
        else:
            scores = torch.stack(scores, dim=0)

        # the separation of beam_scores and logits_for_bce gives us more flexibility in training
        return top_candidates, scores, beam_scores_list, label_list, logits_for_bce, gold_candidates

    def _get_gold_ids(self, gold_programs, candidate_programs, entity_name=None):
        gold_ids = []
        em_gold_ids = []
        if isinstance(gold_programs, list):
            for program in gold_programs:
                if not self._em_augmentation:
                    flag = False
                    for i in range(len(candidate_programs)):
                        if candidate_programs[i].code_raw == program.code_raw:
                            gold_ids.append(i)
                            flag = True
                            break
                else:
                    flag = False
                    for i in range(len(candidate_programs)):
                        if candidate_programs[i].code_raw == program.code_raw:
                            gold_ids.append(i)
                            flag = True
                        elif same_logical_form(postprocess_raw_code(candidate_programs[i].code_raw),
                                               postprocess_raw_code(program.code_raw)):
                            em_gold_ids.append(i)

                if not flag:
                    try:
                        wrapped_program = self._wrap_program(program, entity_name)
                    except Exception:
                        print("wrapped_program exception:", program)
                    if wrapped_program is not None:
                        candidate_programs.append(wrapped_program)
                    gold_ids.append(len(candidate_programs) - 1)
        else:
            if not self._em_augmentation:
                flag = False
                for i in range(len(candidate_programs)):
                    if candidate_programs[i].code_raw == gold_programs.code_raw:
                        gold_ids.append(i)
                        flag = True
                        break
            else:
                flag = False
                for i in range(len(candidate_programs)):
                    if candidate_programs[i].code_raw == gold_programs.code_raw:
                        gold_ids.append(i)
                        flag = True
                    elif same_logical_form(postprocess_raw_code(candidate_programs[i].code_raw),
                                           postprocess_raw_code(gold_programs.code_raw)):
                        em_gold_ids.append(i)

            if not flag:
                # candidate_programs.append(Program(code=gold_programs, code_raw=gold_target, finalized=True))
                wrapped_program = self._wrap_program(gold_programs, entity_name)
                if wrapped_program is not None:
                    candidate_programs.append(wrapped_program)
                gold_ids.append(len(candidate_programs) - 1)

        return gold_ids, em_gold_ids

    def _score_pairs_with_t5(self, pairs: List[Tuple[str, str]], batch_size=256, target='<extra_id_23>'):
        concat = []
        target_id = self._tokenizer(target).input_ids
        assert len(target_id) == 2
        target_id = target_id[0]

        for pair in pairs:
            concat.append(pair[0] + '</s>' + pair[1])

        num_batch = math.ceil(len(pairs) / batch_size)
        logits_list = []
        for i in range(num_batch):
            current_batch = concat[i * batch_size:(i + 1) * batch_size]
            pairs_input = self._tokenizer(current_batch, return_tensors='pt', padding=True)
            input_ids = pairs_input.input_ids.to(self._device)[:, :256]
            attention_mask = pairs_input.attention_mask.to(self._device)[:, :256]
            decoder_input_ids = torch.zeros(len(current_batch), 1).to(self._device).int()
            # (batch_size, 1, config.vocab_size)
            outputs = self._enc_dec_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids,
                                          attention_mask=attention_mask, use_cache=False, return_dict=True)["logits"]

            logits_list.append(outputs[:, 0, target_id])

        logits = torch.cat(logits_list, dim=0)
        return logits

    def _score_pairs(self, pairs: List[Tuple[str, str]], batch_size=128):
        num_batch = math.ceil(len(pairs) / batch_size)
        logits_list = []
        for i in range(num_batch):
            plm_input = self.make_plm_input_hf(pairs[i * batch_size:(i + 1) * batch_size])
            pooler_output = self._source_embedder({"tokens": plm_input})
            logits = self.cls_layer(self._dropout(pooler_output))   # this is the same as XXXForSequenceClassification
            logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)

        return logits

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        # only works for batch size 1
        ids = output_dict['ids']

        all_predicted_lfs = []  # all returned values should be lists following AllenNLP's design
        all_predicted_answers = []
        denotation = []

        predicted_program = output_dict["predictions"]
        if predicted_program is not None:  # no admissible program due to el error
            predicted_lf = predicted_program.code_raw
            if predicted_lf != '':
                try:
                    sparql_query = lisp_to_sparql(predicted_lf)
                    execution = execute_query(sparql_query)
                    denotation.extend(execution)
                except Exception:
                    pass
            else:
                denotation = ["1"]
        else:
            predicted_lf = ''
            denotation = ["1"]

        all_predicted_answers.append(denotation)
        all_predicted_lfs.append(predicted_lf)

        rtn = {}
        rtn['qid'] = ids
        rtn['logical_form'] = all_predicted_lfs
        rtn['answer'] = all_predicted_answers

        return rtn

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # reset is set to be True by default in trainer
        all_metrics: Dict[str, float] = {}
        if not self._infer and self._training_option != 2 and self._val_option != 2:
            all_metrics['Precision'] = self._precision.get_metric(reset)
            all_metrics['Recall'] = self._recall.get_metric(reset)
            all_metrics['F1'] = self._F1.get_metric(reset)
            all_metrics['Acc'] = self._accuracy.get_metric(reset)
        elif self._val_option == 2 and self._training_option != 2:
            if self.training:
                all_metrics['Precision'] = self._precision.get_metric(reset)
                all_metrics['Recall'] = self._recall.get_metric(reset)
                all_metrics['F1'] = self._F1.get_metric(reset)
                all_metrics['Acc'] = self._accuracy.get_metric(reset)
                all_metrics['EM'] = self._exact_match.get_metric(reset)
            else:
                all_metrics['example_count'] = self._exact_match._count
                all_metrics['EM'] = self._exact_match.get_metric(reset)
                all_metrics['EM_iid'] = self._exact_match_iid.get_metric(reset)
                all_metrics['EM_comp'] = self._exact_match_comp.get_metric(reset)
                all_metrics['EM_zero'] = self._exact_match_zero.get_metric(reset)
                all_metrics['F1'] = self._F1.get_metric(reset)
        else:
            all_metrics['example_count'] = self._exact_match._count
            all_metrics['EM'] = self._exact_match.get_metric(reset)
            all_metrics['EM_iid'] = self._exact_match_iid.get_metric(reset)
            all_metrics['EM_comp'] = self._exact_match_comp.get_metric(reset)
            all_metrics['EM_zero'] = self._exact_match_zero.get_metric(reset)
            all_metrics['F1'] = self._F1.get_metric(reset)

        return all_metrics

    def make_plm_input_hf(self, pairs):
        input_ids = []
        for pair in pairs:
            ids0 = self._source_tokenizer(pair[0])['input_ids']
            ids1 = self._source_tokenizer(pair[1])['input_ids']
            pair_ids = self._source_tokenizer.build_inputs_with_special_tokens(ids0[1:-1], ids1[1:-1])
            input_ids.append(pair_ids)

        max_len = max([len(ids) for ids in input_ids])
        if self._EOS == '[SEP]':
            padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
        else:
            padded = [ids + [1] * (max_len - len(ids)) for ids in input_ids]
        rtn = torch.tensor(padded).long()
        rtn = rtn.to(self._device)

        return rtn

    def make_plm_input(self, tokenized_seqs):
        max_len = 0
        for tokenized_source in tokenized_seqs:
            if len(tokenized_source) > max_len:
                max_len = len(tokenized_source)

        if self._EOS == '[SEP]':  # deberta and distilbert are the same as bert
            # [PAD] is 0
            bert_input = torch.zeros(len(tokenized_seqs), max_len, device=self._device)
        else:  # roberta
            # [PAD] is 1
            bert_input = torch.ones(len(tokenized_seqs), max_len, device=self._device)
        for i, tokenized_source in enumerate(tokenized_seqs):
            for j, token in enumerate(tokenized_source):
                #  Here namespace "bert" is consistent with the config for indexer
                # bert_input[i][j] = self.vocab.get_token_index(token.text, namespace="bert")
                bert_input[i][j] = self._source_tokenizer.convert_token_to_id(token.text)

        return bert_input.long()

    def _wrap_program(self, gold_program_i, entity_name):
        """
        Wrap a string into a Program object, this is only used for training
        :param gold_program_i:
        :param entity_name:
        :return:
        """
        try:
            expression = lisp_to_nested_expression(gold_program_i.code_raw)
            height = get_nesting_level(expression) - 1
            function = expression[0]

            if (expression[0] == 'AND' and isinstance(expression[1], str)) or expression[0] == 'COUNT':
                finalized = True
            else:
                finalized = False

            if expression[0] in ['ARGMIN', 'ARGMAX'] and isinstance(expression[1], str):
                last_edge = gold_program_i.code_raw.split(' ')[-1][:-1]
                if last_edge in self._computer._attributes:
                    finalized = True
                    execution = None
                else:  # last_edge should be a relation
                    if last_edge[-4:] == '_inv':
                        execution = self._computer._relation_d[last_edge[:-4]]
                    else:
                        execution = self._computer._relation_r[last_edge]
            else:
                execution = set()
                if not finalized:
                    try:
                        # TODO: here you might get some conversion loss
                        sparql_query = lisp_to_sparql(postprocess_raw_code(gold_program_i.code_raw))
                        execution.update(execute_query(sparql_query))
                    except Exception:
                        pass

            return Program(source=gold_program_i.source,
                           code=gold_program_i.code,
                           code_raw=gold_program_i.code_raw,
                           function=function,
                           height=height,
                           finalized=finalized,
                           derivations=get_derivations_from_lisp(lisp_to_nested_expression(gold_program_i.code_raw)),
                           execution=execution)

        except UnboundLocalError:
            return None
