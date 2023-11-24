import allennlp
import time
import random
import re
import json
import math
import logging
import functools
import numpy as np

from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict
from pathlib import Path

from utils import logic_form_util
from utils.logic_form_util import same_logical_form, lisp_to_sparql, postprocess_raw_code, get_derivations_from_lisp, \
    get_sub_programs, fill_sub_programs, max_count_relations
from utils.sparql_executer import execute_query
from utils.semparse_util import lisp_to_nested_expression, get_nesting_level
from utils.kb_environment import Computer

# from openai_eval.interface import OpenaiEngine
from LLM_prompts.engines import OpenaiEngine
from LLM_prompts.utils import format_prompt
from openai_eval.prompting import templating_one_example
from openai_eval.prompts import manual_prompt

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

path = str(Path(__file__).parent.absolute())


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


class Program:
    def __init__(self,
                 source: Union[Set, str] = None,
                 code: str = '',  # used for PLM classifier
                 code_raw: str = '',  # original code (i.e., code with mids)
                 function: str = None,
                 height: int = -1,
                 execution: Union[Set, str] = None,
                 finalized: bool = False,
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

    def execute(self, kb_engine=None):
        if kb_engine is None:  # for training
            # if True:  # todo: use another flag for this
            if isinstance(self.execution, tuple):
                self.execution = self.execution[0](*self.execution[1:])
        else:
            if not isinstance(self.execution, set):
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
        assert isinstance(self.execution, set) or isinstance(self.execution, list)
        types = kb_engine.get_classes_for_variables(self.execution)
        cvt = True
        for t in types:
            if t not in kb_engine.cvt_types:
                cvt = False
                break
        return cvt

    def __str__(self):
        return self.code_raw


@Model.register("bottom_up_gpt")
class BottomUpParserGPT(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            beam_size=5,
            decoding_steps=5,  # 5 for grail; 4 for graph
            dataset='grail',
            reverse=False,
            dynamic_retrieval=True,
            offline_retrieval=False,
            sample_num=10,
            diverse=False,
            delex=False,
            penalize=0.7  # used to penalize repeated relations
    ) -> None:
        super().__init__(vocab)

        self._max_count = 2000  # I used 2000 for webqsp

        # self._dataset = "webq"
        self._dataset = dataset

        self._beam_size = beam_size
        self._decoding_steps = decoding_steps

        self.linear = nn.Linear(5, 1)

        self._computer = Computer(dataset=self._dataset, llm=True)
        # We only use BottomUpParserGPT for inference
        self._computer.set_training(False)

        self._engine = OpenaiEngine(api_key=[
            # put you openai key here
        ],
            # model="code-davinci-002",
            model="text-davinci-003",
            rate_limit=15)

        self._exact_match = Average()
        self._exact_match_iid = Average()
        self._exact_match_comp = Average()
        self._exact_match_zero = Average()
        self._F1 = Average()

        self._reverse = reverse
        self._delex = delex
        self._pen = penalize

        if not self._reverse:
            with open(path + "/../LLM_prompts/tasks/kbqa/prompt.gold.codex.json") as f:
                self._prompt = json.load(f)
        else:
            with open(path + "/../LLM_prompts/tasks/kbqa/prompt.reverse.codex.json") as f:
                self._prompt = json.load(f)

        self._dr = dynamic_retrieval
        self._offline = offline_retrieval
        self._diverse = diverse
        if not self._dr:
            # with open(path + "/../LLM_prompts/tasks/kbqa/teaching.4-shot.gold.json") as f:
            with open(path + "/../LLM_prompts/tasks/kbqa/teaching_10shot.json") as f:
                # with open(path + "/../LLM_prompts/tasks/teaching_memorization.json") as f:
                # with open(path + "/../LLM_prompts/tasks/kbqa/count_5shot.json") as f:
                self._teaching_examples = json.load(f)[:sample_num]
                for item in self._teaching_examples:  # Sample num is applied here !!!
                    item['query'] = item['query'].lower()
                    item['question'] = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', item['question']).replace(" '",
                                                                                                       "'").lower()

        else:
            self._examples_num = sample_num
            if self._offline:
                # with open(path + "/dynamic_retrieval/grailqa_dev_retrieved_100.json") as f:
                with open(path + "/dynamic_retrieval/grailqa_dev_delex_retrieved.json") as f:
                    self._corpus = json.load(f)
            else:
                from openai_eval.dynamic_retrieval.IRModel import IRBm25
                self._retriever = IRBm25()
                if self._dataset == "grail":
                    # with open(path + "/../data/stratified_samples.json") as f:
                    with open(path + "/../data/grailqa_v1.0_train.json") as f:
                        self._raw_corpus = self.process_corpus(json.load(f)[1:2])
                        # self._raw_corpus = self.process_corpus(json.load(f)[:10])
                        # self._raw_corpus = self.process_corpus(json.load(f)[2000:3000])
                        # self._raw_corpus = self.process_corpus(json.load(f)[:100])
                        # self._raw_corpus = self.process_corpus(json.load(f))
                        print(len(self._raw_corpus))
                        self._corpus = {}
                elif self._dataset == "gq1":
                    # with open(path + "/../data/stratified_samples_gq1.json") as f:
                    with open(path + "/../data/graphquestions_v1_fb15_training_091420.json") as f:
                        training_data = json.load(f)
                        random.shuffle(training_data)
                        # self._raw_corpus = self.process_corpus(training_data[:10])
                        self._raw_corpus = self.process_corpus(training_data[:1000])
                        # print(len(self._raw_corpus))
                        self._corpus = {}
                elif self._dataset == "webq":
                    # with open(path + "/../data/stratified_samples_webq.json") as f:
                    with open(path + "/../data/webqsp_0107.train.json") as f:
                        # self._raw_corpus = self.process_corpus(json.load(f)[:1000])
                        self._raw_corpus = self.process_corpus(json.load(f)[:10])
                        self._corpus = {}

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
        if self.training:  # dummy for allennlp
            self._loss = nn.BCEWithLogitsLoss()
            output_dict = {}
            self._device = next(self.linear.parameters()).device
            inputs = torch.rand(1, 5).to(self._device)
            label = torch.ones(1, 1).to(self._device)
            loss = self._loss(self.linear(inputs), label)
            output_dict["loss"] = loss

            return output_dict

        if self._dr:
            qid = str(ids[0])
            if not self._offline and qid not in self._corpus:  # online retrieval
                query = question[0]
                for k, v in entity_name[0].items():
                    query = query.replace(v.lower(), "[ENT]")
                candidates = self._retriever.get_top_n(question=query, candidates=list(self._raw_corpus.keys()),
                                                       n=self._examples_num, tau=None)
                # print(candidates)
                similar_examples = []
                for c in candidates:
                    similar_examples.extend(self._raw_corpus[c])
                self._corpus[qid] = similar_examples
            self._teaching_examples = []
            if not self._diverse:
                for item in self._corpus[qid][:self._examples_num]:
                    if not self._delex:
                        self._teaching_examples.append({"question": item["question"],
                                                        "query": item["query"]})
                    else:
                        self._teaching_examples.append({"question": item["question_delex"],
                                                        "query": item["query_delex"]})
            else:
                canonical_forms = set()
                for item in self._corpus[qid]:
                    if len(canonical_forms) == self._examples_num:
                        break
                    new_flag = True
                    for cf in canonical_forms:
                        if same_logical_form(cf.replace("[ENT]", "m.123"),
                                             item['query_delex'].replace("[ENT]", "m.123")):
                            new_flag = False
                    # if item["query_delex"] not in canonical_forms:
                    if new_flag:
                        if not self._delex:
                            self._teaching_examples.append({"question": item["question"],
                                                            "query": item["query"]})
                        else:
                            self._teaching_examples.append({"question": item["question_delex"],
                                                            "query": item["query_delex"]})
                        canonical_forms.add(item["query_delex"])

        # print("requests:", self._engine.requests, file=open("./openai_eval/requests_count.txt", 'a'))
        predictions = None

        programs: List[List[List[Program]]] = []
        programs_indexed = [defaultdict(lambda: []) for _ in range(len(question))]  # len batch_size

        # best_candidates = ['' for _ in
        #                    range(len(raw_question))]  # used to track the best candidate for termination check

        highest_scores = [-1e32 for _ in range(len(question))]
        highest_finalized = None
        highest_finalized_score = -1e32

        num_candidates = 0
        for decoding_step in range(self._decoding_steps):
            candidate_programs = []
            if decoding_step == 0:
                for i, en in enumerate(entity_name):
                    ini_programs_i = self._computer.get_initial_programs(en, answer_types[i], gold_answer_type[i])
                    new_ini_programs_i = []
                    for ip in ini_programs_i:
                        # if ip.function in ["AND", "JOIN"]:
                        #     new_ini_programs_i.append(ip)
                        new_ini_programs_i.append(ip)
                    ini_programs_i = new_ini_programs_i
                    if len(ini_programs_i) > self._max_count:  # we can skip these to save some money
                        ini_programs_i = ini_programs_i[:self._max_count]
                        print(ids, file=open('openai_eval/requestout.txt', 'a'))

                    candidate_programs.append(ini_programs_i)
            else:
                for i in range(len(programs_indexed)):  # for i in range(batch_size)
                    candidate_programs_i = self._computer.get_admissible_programs(programs[decoding_step - 1][i],
                                                                                  programs_indexed[i],
                                                                                  entity_name[i]
                                                                                  )
                    # new_candidate_programs_i = []
                    # for ip in candidate_programs_i:
                    #     if ip.function in ["AND", "JOIN"]:
                    #         new_candidate_programs_i.append(ip)
                    # candidate_programs_i = new_candidate_programs_i

                    if len(candidate_programs_i) > self._max_count:  # we can skip these to save some money
                        candidate_programs_i = candidate_programs_i[:self._max_count]
                        print(ids, file=open('openai_eval/requestout.txt', 'a'))

                    candidate_programs.append(candidate_programs_i)
                    num_candidates += len(candidate_programs_i)

            if len(candidate_programs[0]) == 0:  # normally due to all beam programs being finalized
                break
            else:
                # elif len(candidate_programs[0]) > 1 or self.training:
                # print(len(candidate_programs[0]))
                # During training, beam_logits may not be strictly in order for two reasons:
                # 1) Gold ids are manually populated into the beam
                # 2) The scores for beam items are recomputed for backprop with dropout
                # new_beam_programs = self._get_top_candidates_gpt3(candidate_programs, raw_question)
                new_beam_programs, beam_scores = self._get_top_candidates_codex(candidate_programs, question,
                                                                                entity_name)

                # for bp in new_beam_programs[0]:
                #     bp.execute(self._computer)

                termination_flag = False
                for i in range(len(highest_scores)):
                    # todo: for batching, need to handle the asynchronous termination issue
                    if len(beam_scores[i]) > 0 and beam_scores[i][0] > highest_scores[0]:
                        # for inference, scores should be in descending order
                        highest_scores[0] = beam_scores[i][0]
                    elif decoding_step > 0:
                        termination_flag = True
                        break

                no_finalized = True
                for i, beam_cand in enumerate(new_beam_programs[0]):
                    if beam_cand.finalized:
                        no_finalized = False
                        if highest_finalized is None or highest_finalized_score < beam_scores[0][i]:
                            highest_finalized = beam_cand
                            highest_finalized_score = beam_scores[0][i]
                        break

                # if termination_flag or (no_finalized and highest_finalized is not None):
                if termination_flag:
                    break

                # update beam_programs to the current step
                beam_programs = new_beam_programs

            programs.append(beam_programs)
            for i, candidates in enumerate(beam_programs):
                for p in candidates:
                    if isinstance(p.source, set):
                        p.source = tuple(p.source)
                    programs_indexed[i][p.source].append(p)

        try:
            if highest_finalized is not None:
                predictions = highest_finalized
            else:
                finalized = False
                for p in beam_programs[0]:  # only works for batch size 1
                    if p.finalized and (
                            p.execution is None or (isinstance(p.execution, int) and p.execution != 0) or (
                            not isinstance(p.execution, int) and len(p.execution) > 0 and not p.is_cvt(
                        self._computer))):
                        finalized = True
                    predictions = p
                    break
                if not finalized:  # todo: here still need to filter null answer
                    if len(beam_programs[0]) > 0:
                        predictions = beam_programs[0][0]
                    elif len(candidate_programs_i) > 0:
                        predictions = candidate_programs_i[0]  # ideally, this should never happen
                    else:
                        predictions = Program()
                        print("wtf!!!!")

            predictions.code_raw = postprocess_raw_code(predictions.code_raw)
            if gold_program[0] is not None:
                em = same_logical_form(predictions.code_raw, gold_program[0])
            else:
                em = 0
            print("total passes:", num_candidates,
                  file=open(path + f"/{self._dataset}_num_calls_{self._beam_size}.txt", 'a'))
        except UnboundLocalError:  # beam_programs referenced before assignment
            # possible reasons for this:
            # 1. empty entity linking results before implementing superlatives
            # 2. no admissible relations for entities
            em = 0
            print("question:", question)

        self._exact_match(em)
        if level[0] == "i.i.d.":
            self._exact_match_iid(em)
        if level[0] == "compositional":
            self._exact_match_comp(em)
        if level[0] == "zero-shot":
            self._exact_match_zero(em)

        output_dict = {"predictions": predictions,
                       "ids": ids}

        return output_dict

    @DeprecationWarning
    def _get_top_candidates_gpt3(self, candidate_programs: List[List[Program]],
                                 question: List[str]
                                 ):
        if len(candidate_programs[0]) <= self._beam_size:
            return [candidate_programs[0][:]]
        # todo: implement min heap to return top-K here
        beam_programs = [[]]
        for i in range(self._beam_size):
            best_id = 0
            for j in range(len(candidate_programs[0]) - 1):
                if self._get_label_gpt3(question[0], candidate_programs[0][j].code,
                                        candidate_programs[0][best_id].code) == "A":
                    best_id = j
            beam_programs[0].append(candidate_programs[0][best_id])
            candidate_programs[0].pop(best_id)

        return beam_programs

    @DeprecationWarning
    def _get_label_gpt3(self, question, program_a, program_b):
        # print(question)
        # print(program_a)
        # print(program_b)
        prompt = manual_prompt + '\n' + templating_one_example(question, program_a, program_b, '')
        response = self._engine.generate(model="text-davinci-002", prompts=prompt, max_new_tokens=2, temperature=0, n=1)
        self._engine.requests += 1
        return response['outputs'][0][1]

    def _get_top_candidates_codex(self, candidate_programs: List[List[Program]],
                                  question: List[str],
                                  entity_name
                                  ):
        if self._delex:
            for cand in candidate_programs[0]:
                cand.code = cand.code_raw
            for k, v in entity_name[0].items():
                question[0] = question[0].replace(v.lower(), "[ENT]")
                for cand in candidate_programs[0]:
                    cand.code = cand.code.replace(k, "[ENT]")

        scores = self._score_pairs_codex(question[0], candidate_programs[0])
        indices = np.argsort(scores)[::-1]
        top_indices = indices[:10]

        beam_candidates_i = []
        scores_i = []
        for i, idx in enumerate(top_indices):
            if len(beam_candidates_i) == self._beam_size:
                break
            candi = candidate_programs[0][idx]
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
            beam_candidates_i.append(candi)
            scores_i.append(scores[idx])

        beam_programs = [beam_candidates_i]
        beam_scores = [scores_i]

        return beam_programs, beam_scores

    def _score_pairs_codex(self, question, candidates, batch_size=20):
        if not self._delex:
            question = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', question).replace(" '", "'")
            question = question.lower()
        scores = []
        num_batch = math.ceil(len(candidates) / batch_size)
        for b in range(num_batch):
            formatted_prompts = []
            targets = []
            pen_factors = [1.0] * len(candidates[b * batch_size:(b + 1) * batch_size])
            for i, cand in enumerate(candidates[b * batch_size:(b + 1) * batch_size]):
                if not self._delex:
                    example = {'question': question,
                               # 'query': cand.code.lower()}
                               'query': cand.code}
                else:
                    example = {'question': question,
                               'query': cand.code}
                # self._teaching_examples = []
                formatted_prompt = format_prompt(self._prompt, self._teaching_examples, example)
                # print(formatted_prompt)
                # formatted_prompt = formatted_prompt.strip()
                formatted_prompts.append(formatted_prompt)
                if not self._reverse:
                    # targets.append(cand.code.lower())  # todo: why did I do lower here???
                    targets.append(cand.code)
                else:
                    targets.append(question)

                if self._pen is not None:
                    relation_count = max_count_relations(postprocess_raw_code(cand.code_raw))
                    if relation_count > 1:
                        pen_factors[i] = math.pow(self._pen, relation_count - 1)


            responses = self._engine.score(formatted_prompts, targets)
            for i, response in enumerate(responses):
                assert response['outputs'][0] == response['raw'][0]['sequence_logprob']
                # scores.append(response['outputs'][0])
                scores.append(response['outputs'][0] / pen_factors[i])
                # example.update({"uid": uid, "response": response})
        assert len(candidates) == len(scores)
        return scores

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
            try:
                sparql_query = lisp_to_sparql(predicted_lf)
                execution = execute_query(sparql_query)
                denotation.extend(execution)
            except Exception:
                pass
        else:
            predicted_lf = ''

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

        if not self.training:
            all_metrics['example_count'] = self._exact_match._count
            all_metrics['EM'] = self._exact_match.get_metric(reset)
            all_metrics['EM_iid'] = self._exact_match_iid.get_metric(reset)
            all_metrics['EM_comp'] = self._exact_match_comp.get_metric(reset)
            all_metrics['EM_zero'] = self._exact_match_zero.get_metric(reset)
            all_metrics['F1'] = self._F1.get_metric(reset)

        return all_metrics

    def process_corpus(self, data):
        corpus = defaultdict(lambda: [])
        for item in data:
            if item["s_expression"] is None:
                continue
            entity_name_map = {}
            entity_name_map_delex = {}
            question = item['question']
            for node in item['graph_query']['nodes']:
                if node['node_type'] in ['entity', 'literal'] and node['function'] not in ['argmax', 'argmin']:
                    entity_name_map[node['id']] = node['friendly_name'].lower()
                    entity_name_map_delex[node['id']] = '[ENT]'
                    question = question.replace(node['friendly_name'].lower(), '[ENT]')
            gold_sub_programs, level_mapping = get_sub_programs(item["s_expression"])
            gold_sub_programs_filled = fill_sub_programs(gold_sub_programs, entity_name_map)
            # gold_sub_programs, level_mapping = get_sub_programs(item["s_expression"])
            gold_sub_programs_filled_delex = fill_sub_programs(gold_sub_programs, entity_name_map_delex)
            # print(item['s_expression'])
            processed_expression = gold_sub_programs_filled[-1]
            processed_expression_delex = gold_sub_programs_filled_delex[-1]
            corpus[question].append({"qid": item['qid'],
                                     "question": item['question'],
                                     "query": processed_expression,
                                     "question_delex": question,
                                     "query_delex": processed_expression_delex})

        return corpus
