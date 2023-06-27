import random
import re
import time
import functools
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from utils.logic_form_util import lisp_to_sparql, postprocess_raw_code
from utils.semparse_util import lisp_to_nested_expression
from utils.sparql_cache import SparqlCache
from new_model.bottom_up_parser import Program

path = str(Path(__file__).parent.absolute())


# todo: handle the conversion loss of lisp-to-sparql
# todo: for superlatives, now we are unable to use get XXX for program. handle it later

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


def get_vocab(dataset: str):
    if dataset == "grail":
        with open(path + '/vocab_files/grailqa.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"])
    elif dataset == "gq1":
        with open(path + '/vocab_files/gq1.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"])
    elif dataset == "webq":
        # with open(path + '/vocab_files/webq.json') as f:
        with open(path + '/vocab_files/webq_full.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"]), set(data["tc_attributes"]), set(
            data["cons_attributes"]), data["cons_ids"]
    elif dataset == "cwq":
        pass


def get_ontology(dataset: str):
    class_hierarchy = defaultdict(lambda: [])
    class_out_edges = defaultdict(lambda: set())
    class_in_edges = defaultdict(lambda: set())
    relation_domain = {}
    relation_range = {}
    date_attributes = set()
    numerical_attributes = set()
    if dataset == "grail":
        fb_type_file = path + "/../ontology/commons/fb_types"
        fb_roles_file = path + "/../ontology/commons/fb_roles"
    elif dataset == "gq1":
        fb_type_file = path + "/../ontology/fb_types"
        fb_roles_file = path + "/../ontology/fb_roles"

    else:  # webq does not need these information
        return class_out_edges, class_in_edges, relation_domain, relation_range, date_attributes, numerical_attributes

    with open(fb_type_file) as f:
        for line in f:
            fields = line.split()
            if fields[2] != "common.topic":
                class_hierarchy[fields[0]].append(fields[2])

    with open(fb_roles_file) as f:
        for line in f:
            fields = line.split()
            relation_domain[fields[1]] = fields[0]
            relation_range[fields[1]] = fields[2]

            class_out_edges[fields[0]].add(fields[1])
            class_in_edges[fields[2]].add(fields[1])

            if fields[2] in ['type.int', 'type.float']:
                numerical_attributes.add(fields[1])
            elif fields[2] == 'type.datetime':
                date_attributes.add(fields[1])

    for c in class_hierarchy:
        for c_p in class_hierarchy[c]:
            class_out_edges[c].update(class_out_edges[c_p])
            class_in_edges[c].update(class_in_edges[c_p])

    return class_out_edges, class_in_edges, relation_domain, relation_range, date_attributes, numerical_attributes


def _extend_deri(derivations, source, relation):
    new_derivations = {}
    if isinstance(source, str):
        if isinstance(derivations[source], list):
            new_derivations[source] = derivations[source][:]
            new_derivations[source].extend([':' + relation[:-4] if relation[-4:] == '_inv' else '^:' + relation])
        elif isinstance(derivations[source], tuple):
            new_paths = derivations[source][0][:]
            new_paths.extend([':' + relation[:-4] if relation[-4:] == '_inv' else '^:' + relation])
            new_derivations[source] = (
                new_paths,
                derivations[source][1])

    return new_derivations


class Computer:
    def __init__(self, dataset='grail', MAX_VARIABLES_NUM=20, llm=False):
        self._dataset = dataset
        self._llm = llm
        if dataset in ["grail", "gq1"]:
            self._relations, self._classes, self._attributes = get_vocab(dataset)
        elif dataset == "webq":
            self._relations, self._classes, self._attributes, self._tc_attributes, self._cons_attributes, self._cons_ids = get_vocab(
                dataset)
            if self._llm:
                with open(path + "/webqsp_schema_train.json") as f:
                    webqsp_schema = set(json.load(f))
                self._relations = webqsp_schema.intersection(self._relations)
                self._attributes = webqsp_schema.intersection(self._attributes)


        if dataset == "grail":
            with open('ontology/domain_dict', 'r') as f:
                self._domain_dict = json.load(f)
            with open('ontology/domain_info', 'r') as f:
                self._domain_info = json.load(f)
        self._class_out, self._class_in, self._relation_d, self._relation_r, self._date_attributes, \
        self._numerical_attributes = get_ontology(dataset)
        self._date_attributes = self._date_attributes.intersection(self._attributes)
        self._numerical_attributes = self._numerical_attributes.intersection(self._attributes)
        self._cache = SparqlCache(dataset)
        self.training = False
        self.max_variables_num = MAX_VARIABLES_NUM

        self.cvt_types = set()
        with open(path + "/../ontology/cvt_types.txt") as f:
            for line in f:
                self.cvt_types.add(line.replace('\n', ''))

    def get_vocab(self):
        return self._relations, self._classes, self._attributes

    def set_training(self, training):  # call it at the beginning of each forward pass
        self.training = training

    def process_value(self, value):
        data_type = value.split("^^")[1].split("#")[1]
        if data_type not in ['integer', 'float', 'double', 'dateTime']:
            value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
        else:
            value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

        return value

    def get_relations_for_program(self, program, reverse=False):
        if self.training:
        # if True:  # todo: use another flag for this
            results = self.get_relations_for_variables(program.execution, reverse=reverse)
        else:
            processed_code = postprocess_raw_code(program.code_raw)
            sparql_query = lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            if reverse:
                new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
            else:
                new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?sub ?rel ?x .\n{"]
            new_clauses.extend(clauses[1:])
            new_clauses.append("}\n}")

            new_query = '\n'.join(new_clauses)
            try:
                results = self.execute_SPARQL(new_query)
            except Exception:
                results = self.get_relations_for_variables(program.execution, reverse=reverse)

        rtn = results.intersection(self._relations)

        return rtn

    def get_relations_for_variables(self, entities, reverse=False, add_noise=False):
        '''
        The most straightforward way is obviously get those relations using SPARQL query, but I am not sure about
        the efficiency of doing this.
        Also, for debug purpose, we can also just simply return all the relations in Freebase to make sure the whole
        flow works.
        :param entities: A set of entities
        :param reverse: True indicates outgoing relations, while False indicates ingoing relations
        :return: All adjacent relations of those entities
        '''

        # if TC:
        #     tc_relations = set()
        #     for r in self._relations:
        #         if r.__contains__(".from"):
        #             tc_relations.add(r)
        #     return tc_relations

        # print("get relations for: {} entities".format(len(entities)))
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:100]:
            try:
                if reverse:
                    rtn.update(self._cache.get_out_relations(entity).intersection(self._relations))
                else:
                    rtn.update(self._cache.get_in_relations(entity).intersection(self._relations))
            except Exception:
                # print("entity:", entity)
                pass
        # print(entities)
        # print("done getting relations")

        if self.training and add_noise:
            if not self._dataset == 'grail':
                rtn.update(random.sample(self._relations, 100))
            elif len(self._domains) > 0:
                if random.random() > 0.5:
                    vocab = set()
                    for d in self._domains:
                        vocab.update(self._domain_dict[d])
                    # rtn = rtn.intersection(vocab)
                    if len(vocab) > 100:
                        rtn.update(random.sample(vocab, 100))
                    else:
                        rtn.update(vocab)

        return rtn

    def get_relations_for_class(self, class_name, reverse=False, add_noise=False):
        if reverse:
            return self._class_out[class_name].intersection(self._relations)
        else:
            return self._class_in[class_name].intersection(self._relations)

    def get_attributes_for_program(self, program):
        if self.training:
        # if True:  # todo: use another flag for this
            results = self.get_attributes_for_variables(program.execution)
        else:
            processed_code = postprocess_raw_code(program.code_raw)
            sparql_query = lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            new_clauses = [clauses[0], "SELECT DISTINCT ?att\nWHERE {\n?x ?att ?obj .\n{"]
            new_clauses.extend(clauses[1:])
            new_clauses.append("}\n}")

            new_query = '\n'.join(new_clauses)
            try:
                results = self.execute_SPARQL(new_query)
            except Exception:
                results = self.get_attributes_for_variables(program.execution)

        rtn = results.intersection(self._attributes)

        return rtn

    def get_constraints_for_program(self, program):
        try:
            processed_code = postprocess_raw_code(program.code_raw)
            sparql_query = lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            rtn = set()  # pairs of cons_att and cons_id
            for cons_id in self._cons_ids:
                if cons_id[:2] in ['m.', 'g.']:
                    new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ns:" + cons_id + " .\n{"]
                    new_clauses.extend(clauses[1:])
                    new_clauses.append("}\n}")
                else:  # e.g., State, Country
                    new_clauses = [clauses[0],
                                   "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n FILTER (str(?obj) = \""
                                   + ' '.join(cons_id.split('_')) + "\")\n{"]

                new_query = '\n'.join(new_clauses)
                try:
                    attributes = self.execute_SPARQL(new_query)
                    attributes = attributes.intersection(self._cons_attributes)
                    for att in attributes:
                        rtn.add((att, cons_id, self._cons_ids[cons_id]))
                except Exception:
                    pass

            return rtn
        except Exception:
            print("get constraints error:", program, file=open("logs/websp_error.txt", 'a'))
            return set()

    def get_tc_constraints_for_program(self, program):
        if self.training:
            results = self.get_tc_attributes_for_variables(program.execution)
        else:
            processed_code = postprocess_raw_code(program.code_raw)
            sparql_query = lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            for cons_id in self._cons_ids:
                if cons_id[:2] in ['m.', 'g.']:
                    new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
                    new_clauses.extend(clauses[1:])
                    new_clauses.append("}\n}")
                else:  # e.g., State, Country
                    new_clauses = [clauses[0],
                                   "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n FILTER (str(?obj) = \""
                                   + cons_id + "\")\n{"]

                new_query = '\n'.join(new_clauses)
                try:
                    results = self.execute_SPARQL(new_query)
                except Exception:
                    results = set()

        rtn = results.intersection(self._tc_attributes)
        return rtn

    def get_attributes_for_variables(self, entities, add_noise=False):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:100]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._attributes))
            except Exception:
                # print("entity:", entity)
                pass
        # print(entities)
        # print("done getting relations")

        if self.training and add_noise:
            if len(self._attributes) > 100:
                rtn.update(random.sample(self._attributes, 100))
            else:
                rtn.update(self._attributes)

        return rtn

    def get_tc_attributes_for_variables(self, entities, add_noise=False):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:100]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._tc_attributes))
            except Exception:
                # print("entity:", entity)
                pass

        if self.training and add_noise:
            if len(self._tc_attributes) > 100:
                rtn.update(random.sample(self._tc_attributes, 100))
            else:
                rtn.update(self._tc_attributes)

        return rtn

    def get_cons_attributes_for_variables(self, entities, add_noise=False):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:100]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._cons_attributes))
            except Exception:
                # print("entity:", entity)
                pass

        if self.training and add_noise:
            if len(self._cons_attributes) > 100:
                rtn.update(random.sample(self._cons_attributes, 100))
            else:
                rtn.update(self._cons_attributes)

        return rtn

    def get_attributes_for_value(self, value, add_noise=False, use_ontology=False):
        rtn = set()

        if use_ontology:
            if value.__contains__("#float") or value.__contains__("#integer") or value.__contains__("#double"):
                rtn.update(self._numerical_attributes)
            else:
                rtn.update(self._date_attributes)
        else:  # retrieve based on KB facts
            data_type = value.split("#")[1]
            if data_type not in ['integer', 'float', 'double', 'dateTime']:
                value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            else:
                value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

            rtn.update(self._cache.get_in_attributes(value).intersection(self._attributes))

        if self.training and add_noise:
            if len(self._attributes) > 100:
                rtn.update(random.sample(self._attributes, 100))
            else:
                rtn.update(self._attributes)

        return rtn

    def get_attributes_for_class(self, class_name, add_noise=False):
        return self._class_out[class_name].intersection(self._attributes)

    def is_intersectant(self, derivation1, derivation2):
        return self._cache.is_intersectant(derivation1, derivation2)

    def get_reachable_classes(self, derivations, answer_types):
        reachable_classes = set()
        for a in answer_types:
            flag = True
            for d in derivations:
                if d[:2] in ['m.', 'g.']:
                    source = ':' + d
                else:
                    source = self.process_value(d)
                if isinstance(derivations[d], list):
                    derivation = [source, derivations[d]]
                elif isinstance(derivations[d], tuple):
                    if derivations[d][1] == 'ge':
                        comp = '>='
                    elif derivations[d][1] == 'gt':
                        comp = '>'
                    elif derivations[d][1] == 'le':
                        comp = '<='
                    elif derivations[d][1] == 'lt':
                        comp = '<'
                    derivation = [source, derivations[d][0], comp]
                if not self._cache.is_reachable(derivation, a):
                    flag = False
                    break
            if flag:
                reachable_classes.add(a)

        return reachable_classes

    def get_classes_for_program(self, program):
        if self.training:
        # if True:  # todo: use another flag for this
            results = self.get_classes_for_variables(program.execution)
        else:
            processed_code = postprocess_raw_code(program.code_raw)
            sparql_query = lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            new_clauses = [clauses[0], "SELECT DISTINCT ?cls\nWHERE {\n?x ns:type.object.type ?cls .\n{"]
            new_clauses.extend(clauses[1:])
            new_clauses.append("}\n}")

            new_query = '\n'.join(new_clauses)
            try:
                results = self.execute_SPARQL(new_query)
            except Exception:
                results = self.get_classes_for_variables(program.execution)

        rtn = results.intersection(self._classes)

        return rtn

    def get_classes_for_variables(self, entities, add_noise=False, cvt_check=False):
        # print("get classes for: {} entities".format(len(entities)))
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:100]:
            if cvt_check and self._dataset == 'webq':
                rtn.update(self._cache.get_types(entity))
            else:
                rtn.update(set(self._cache.get_types(entity)).intersection(self._classes))

        if self.training and add_noise:
            if not self._dataset == "grail":
                if len(self._classes) > 100:
                    rtn.update(random.sample(self._classes, 100))
                else:
                    rtn.update(self._classes)
            elif len(self._domains) > 0:
                if random.random() > 0.5:
                    vocab = set()
                    for d in self._domains:
                        vocab.update(self._domain_dict[d])
                    # rtn = rtn.intersection(vocab)
                    if len(vocab) > 100:
                        rtn.update(random.sample(vocab, 100))
                    else:
                        rtn.update(vocab)

        return rtn

        # return classes

    def get_constraints_for_variables(self, entities, cons_attribute):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:100]:
            rtn.update(set(self._cache.get_out_entities(entity, cons_attribute)).intersection(self._cons_ids))

        return rtn

    def execute_SPARQL(self, sparql_query):
        rtn = self._cache.get_sparql_execution(sparql_query)

        return set(rtn)

    def execute_AND(self, arg1, arg2):
        if not isinstance(arg2, set):
            rtn = set()
            # TODO: this is only for debug
            for entity in list(arg1)[:100]:
                if arg2 in self._cache.get_types(entity):
                    rtn.add(entity)
            return rtn
        else:
            return arg1.intersection(arg2)

    def execute_COUNT(self, arg1):
        return len(arg1)

    def execute_JOIN(self, arg1, arg2):
        # print("execute JOIN for: {} entities".format(len(arg1)))
        rtn = set()
        if isinstance(arg1, str):
            value = arg1
            data_type = value.split("^^")[1].split("#")[1]
            if data_type not in ['integer', 'float', 'double', 'dateTime']:
                value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
                # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
            else:
                value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

            rtn.update(self._cache.get_in_entities_for_literal(value, arg2))
        else:
            if arg2[-4:] == '_inv':
                # TODO: this is only for debug
                for entity in list(arg1)[:100]:
                    # print(entity, arg2[1])
                    rtn.update(self._cache.get_out_entities(entity, arg2[:-4]))
            else:
                # TODO: this is only for debug
                for entity in list(arg1)[:100]:
                    # print(arg2, entity)
                    rtn.update(self._cache.get_in_entities(entity, arg2))
        # print("done executing JOIN")
        return rtn

    def execute_TC(self, arg1, arg2, arg3):
        # TODO: apply time constraint (not urgent)
        return arg1

    def execute_Comparative(self, arg1, arg2, comparator):
        assert isinstance(arg1, str)  # it must be a value instead of a set of entities
        value = arg1
        if comparator == 'le':
            comp = '<='
        elif comparator == 'lt':
            comp = '<'
        elif comparator == 'ge':
            comp = '>='
        elif comparator == 'gt':
            comp = '>'

        data_type = value.split("^^")[1].split("#")[1]
        if data_type not in ['integer', 'float', 'double', 'dateTime']:
            value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
        else:
            value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

        rtn = set()
        rtn.update(self._cache.get_entities_cmp(value, arg2, comp))

        return rtn

    # @timer
    def get_admissible_programs(self, programs: List[Program],
                                programs_indexed: Dict[str, List[Program]],
                                entity_name=None):
        """
        Given beam programs of the current decoding step, return all possible candidate programs of next step.
        It only handles one batch instance.
        :param programs: stores the history of the beam, where each item denotes a list of programs of a certain height
        :param programs_indexed: key: source, value: List of corresponding programs
        :param entity_name: only needed by webqsp; for constraints and tc constraints
        :return: all possible programs of height+1
        """

        # TODO: use direct queries to get candidate relations/types!!! Querying from executions is inefficient

        global program
        candidate_programs = []

        for program in programs:
            if program.dummy:
                continue
            expression = lisp_to_nested_expression(program.code_raw)
            if (program.function == 'AND' and isinstance(expression[1], str)) \
                    or program.function == 'COUNT':
                # todo: after fixing the uncovered vocab items issue, uncomment this assertion
                # assert expression[1] in self._classes
                # pass  # this is a finalized program
                # candidate_programs.append(program)  # pass it to the next step

                # add superlatives and count
                if program.function == 'AND':
                    # handle count here
                    code_i = f'(COUNT {program.code})'
                    code_raw_i = f'(COUNT {program.code_raw})'
                    candidate_programs.append(Program(source=program.source,
                                                      code=code_i,
                                                      code_raw=code_raw_i,
                                                      function='COUNT',
                                                      derivations=program.derivations,
                                                      height=program.height + 1,
                                                      execution=(self.execute_COUNT, program.execution),
                                                      finalized=True))

                    # handle superlatives
                    # possible_relations = self.get_relations_for_variables(program.execution, reverse=True)
                    possible_relations = self.get_relations_for_program(program, reverse=True)
                    possible_relations.update(map(lambda x: x + '_inv',
                                                  # self.get_relations_for_variables(program.execution,
                                                  self.get_relations_for_program(program,
                                                                                 reverse=False)))
                    # TODO: check whether the chain of relations in superlatives in handled properly (i.e., preprocessing datareader and post processing after inference)
                    for r in possible_relations:
                        for func in ['ARGMAX', 'ARGMIN']:
                            code_i = f'({func} {program.code} {r})'
                            code_raw_i = f'({func} {program.code_raw} {r})'

                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function=func,
                                                              # In arg mode, derivations should mostly be useless
                                                              # derivations=_extend_deri(program.derivations,
                                                              #                          program.source, r),
                                                              height=program.height + 1,
                                                              execution=(self.execute_JOIN, program.execution, r)
                                                              ))

                    # possible_attributes = self.get_attributes_for_variables(program.execution)
                    possible_attributes = self.get_attributes_for_program(program)
                    for a in possible_attributes:
                        for func in ['ARGMAX', 'ARGMIN']:
                            code_i = f'({func} {program.code} {a})'
                            code_raw_i = f'({func} {program.code_raw} {a})'
                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function=func,
                                                              # In arg mode, derivations should mostly be useless
                                                              # derivations=_extend_deri(program.derivations,
                                                              #                          program.source, a),
                                                              height=program.height + 1,
                                                              finalized=True,
                                                              # execution=(self.execute_JOIN, program.execution, a)
                                                              ))

            elif program.function in ['ARGMAX', 'ARGMIN']:
                if program.finalized:
                    # candidate_programs.append(program)
                    pass
                else:
                    if isinstance(program.execution, str):  # arg class
                        assert self._dataset != "webq"
                        arg_class = program.execution
                        possible_relations = self.get_relations_for_class(arg_class, reverse=True)
                        possible_relations.update(map(lambda x: x + '_inv',
                                                      self.get_relations_for_class(arg_class, reverse=False)))
                        for r in possible_relations:
                            if r[-4:] == '_inv':
                                execution = self._relation_d[r[:-4]]
                            else:
                                execution = self._relation_r[r]
                            # extend it by the new relation. For arg class, there should only be one ')'
                            code = program.code.replace(')', f' {r})')
                            code_raw = program.code.replace(')', f' {r})')
                            candidate_programs.append(Program(source=program.source,
                                                              code=code,
                                                              code_raw=code_raw,
                                                              function=program.function,
                                                              # derivations={program.source: program.derivations[
                                                              #     program.source].append(
                                                              #     ':' + r[:-4] if r[-4:] == '_inv' else '^:' + r)},
                                                              height=program.height + 1,
                                                              execution=execution))

                        possible_attributes = self.get_attributes_for_class(arg_class)
                        for a in possible_attributes:
                            code = program.code.replace(')', f' {a})')
                            code_raw = program.code.replace(')', f' {a})')
                            candidate_programs.append(Program(source=program.source,
                                                              code=code,
                                                              code_raw=code_raw,
                                                              function=program.function,
                                                              # derivations={program.source: program.derivations[
                                                              #     program.source].append(':' + a)},
                                                              height=program.height + 1,
                                                              # execution=(self.execute_JOIN, {v}, r),
                                                              finalized=True))

                    elif isinstance(program.execution, set):  # arg variable
                        possible_relations = self.get_relations_for_variables(program.execution, reverse=True)
                        possible_relations.update(map(lambda x: x + '_inv',
                                                      self.get_relations_for_variables(program.execution,
                                                                                       reverse=False)))
                        for r in possible_relations:
                            code_i = program.code[:-1] + f' {r})'
                            code_raw_i = program.code_raw[:-1] + f' {r})'
                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function=program.function,
                                                              # In arg mode, derivations should mostly be useless
                                                              # derivations=_extend_deri(program.derivations,
                                                              #                          program.source, r),
                                                              height=program.height + 1,
                                                              execution=(self.execute_JOIN, program.execution, r)))

                        possible_attributes = self.get_attributes_for_variables(program.execution)
                        for a in possible_attributes:
                            code_i = program.code[:-1] + f' {a})'
                            code_raw_i = program.code_raw[:-1] + f' {a})'
                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function=program.function,
                                                              # In arg mode, derivations should mostly be useless
                                                              # derivations=_extend_deri(program.derivations,
                                                              #                          program.source, a),
                                                              height=program.height + 1,
                                                              finalized=True,
                                                              # execution=(self.execute_JOIN, program.execution, a)
                                                              ))


            else:
                # possible_relations = self.get_relations_for_variables(program.execution)
                possible_relations = self.get_relations_for_program(program)
                # possible_relations.update(map(lambda x: x + '_inv',
                #                               self.get_relations_for_variables(program.execution, reverse=True)))
                possible_relations.update(map(lambda x: x + '_inv',
                                              self.get_relations_for_program(program, reverse=True)))
                for r in possible_relations:
                    code_i = f'(JOIN {r} {program.code})'
                    code_raw_i = f'(JOIN {r} {program.code_raw})'
                    # execution = self.execute_JOIN(program.execution, r)
                    if program.derivations is None:
                        print("derivations none:", program.code_raw)

                    candidate_programs.append(Program(source=program.source,
                                                      code=code_i,
                                                      code_raw=code_raw_i,
                                                      function='JOIN',
                                                      derivations=_extend_deri(program.derivations,
                                                                               program.source, r),
                                                      height=program.height + 1,
                                                      finalized=False if self._dataset != 'webq' else True,
                                                      execution=(self.execute_JOIN, program.execution, r)))
                if self.training:
                    if self._dataset != 'webq':
                        possible_types = self.get_classes_for_variables(program.execution)
                    else:
                        possible_types = self._classes
                else:
                    if self._dataset != 'webq':
                        possible_types = self.get_classes_for_program(program)
                    else:
                        try:
                            possible_types = self.get_reachable_classes(program.derivations, self._classes)
                        except Exception:
                            possible_types = []
                            print("get classes error:", program.code_raw, program.derivations,
                                  file=open(path + "/../logs/websp_error.txt", 'a'))
                # # not using previously predicted answer types
                # possible_types = self.get_classes_for_variables(program.execution)
                # possible_types = self.get_classes_for_program(program)
                for t in possible_types:
                    if t in self.cvt_types:
                        continue
                    code_i = f'(AND {t} {program.code})'
                    code_raw_i = f'(AND {t} {program.code_raw})'
                    # execution = self.execute_AND(program.execution, t)
                    candidate_programs.append(Program(source=program.source,
                                                      code=code_i,
                                                      code_raw=code_raw_i,
                                                      function='AND',
                                                      derivations=program.derivations,
                                                      height=program.height + 1,
                                                      execution=(self.execute_AND, program.execution, t),
                                                      finalized=True))

                # AND two subprograms
                # todo: check whether the condition violates some corner case
                if program.function not in ['AND', 'ARGMAX', 'ARGMIN'] and not isinstance(program.source, tuple):
                    for k in programs_indexed:
                        if k != program.source and not isinstance(k, tuple):
                            for p in programs_indexed[k]:
                                if self._dataset != 'webq':
                                    if p.finalized or p.function in ['ARGMAX', 'ARGMIN']:
                                        continue
                                else:
                                    if p.function in ['ARGMAX', 'ARGMIN']:
                                        continue
                                try:
                                    intersection = program.execution.intersection(p.execution)
                                except TypeError:
                                    intersection = set()
                                    print(program.code_raw, program.execution)
                                    print(p.code_raw, p.execution)
                                if len(intersection) > 0:
                                    assert k == p.source
                                    code = f'(AND {program.code} {p.code})'
                                    code_raw = f'(AND {program.code_raw} {p.code_raw})'
                                    execution = intersection
                                    new_derivations = {}
                                    new_derivations.update(p.derivations)
                                    new_derivations.update(program.derivations)
                                    candidate_programs.append(Program(source={k, program.source},
                                                                      code=code,
                                                                      code_raw=code_raw,
                                                                      function='AND',
                                                                      derivations=new_derivations,
                                                                      height=program.height + 1,
                                                                      execution=execution))

                if self._dataset == 'webq':
                    possible_relations = self.get_relations_for_program(program, reverse=True)
                    possible_relations.update(map(lambda x: x + '_inv',
                                                  # self.get_relations_for_variables(program.execution,
                                                  self.get_relations_for_program(program,
                                                                                 reverse=False)))
                    # TODO: check whether the chain of relations in superlatives in handled properly (i.e., preprocessing datareader and post processing after inference)
                    for r in possible_relations:
                        for func in ['ARGMAX', 'ARGMIN']:
                            code_i = f'({func} {program.code} {r})'
                            code_raw_i = f'({func} {program.code_raw} {r})'

                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function=func,
                                                              # In arg mode, derivations should mostly be useless
                                                              # derivations=_extend_deri(program.derivations,
                                                              #                          program.source, r),
                                                              height=program.height + 1,
                                                              execution=(self.execute_JOIN, program.execution, r)
                                                              ))

                    # possible_attributes = self.get_attributes_for_variables(program.execution)
                    possible_attributes = self.get_attributes_for_program(program)
                    for a in possible_attributes:
                        for func in ['ARGMAX', 'ARGMIN']:
                            code_i = f'({func} {program.code} {a})'
                            code_raw_i = f'({func} {program.code_raw} {a})'
                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function=func,
                                                              # In arg mode, derivations should mostly be useless
                                                              # derivations=_extend_deri(program.derivations,
                                                              #                          program.source, a),
                                                              height=program.height + 1,
                                                              finalized=True,
                                                              # execution=(self.execute_JOIN, program.execution, a)
                                                              ))


                    constraints = self.get_constraints_for_program(program)
                    for cons in constraints:
                        code = f'(AND {program.code} (JOIN {cons[0]} {cons[2]}))'
                        code_raw = f'(AND {program.code_raw} (JOIN {cons[0]} {cons[1]}))'
                        candidate_programs.append(Program(source=program.source,
                                                          code=code,
                                                          code_raw=code_raw,
                                                          function='AND',
                                                          derivations=program.derivations,
                                                          height=program.height + 1,
                                                          execution=program.execution,
                                                          finalized=True))

                    tc_constraints = self.get_tc_constraints_for_program(program)
                    for tc_cons in tc_constraints:
                        codes = [f'(TC {program.code} {tc_cons} now)']
                        codes_raw = [f'(TC {program.code_raw} {tc_cons} NOW)']
                        for e in entity_name:
                            if re.match("[\d]{4}", e):
                                code = f'(TC {program.code} {tc_cons} {e})'
                                code_raw = f'(TC {program.code_raw} {tc_cons} {entity_name[e]})'
                                codes.append(code)
                                codes_raw.append(code_raw)

                        for code_i, code_raw_i in zip(codes, codes_raw):
                            candidate_programs.append(Program(source=program.source,
                                                              code=code_i,
                                                              code_raw=code_raw_i,
                                                              function='TC',
                                                              derivations=program.derivations,
                                                              height=program.height + 1,
                                                              execution=program.execution,
                                                              finalized=False))
                            # TC is typically applied to CVT, so finalized is False

        return candidate_programs

    # @timer
    def get_initial_programs(self, entity_name, answer_types, gold_answer_type):
        if answer_types is None and self.training:
            # todo: sample hard negatives
            if self._dataset == 'gq1':
                answer_types = set(random.sample(self._classes, 5))
                answer_types.add(gold_answer_type)
            elif self._dataset == 'grail':
                at_domain = self._domain_info[gold_answer_type]
                domain_types = set(self._domain_dict[at_domain]).intersection(self._classes)
                if len(domain_types) > 5:
                    answer_types = set(random.sample(domain_types, 5))
                else:
                    answer_types = domain_types
                answer_types.add(gold_answer_type)
        initial_programs = []
        for v in entity_name:
            if v[:2] in ['m.', 'g.']:
                possible_relations = self.get_relations_for_variables({v})
                possible_relations.update(map(lambda x: x + '_inv',
                                              self.get_relations_for_variables({v}, reverse=True)))
                for r in possible_relations:
                    code = f'(JOIN {r} {entity_name[v]})'
                    code_raw = f'(JOIN {r} {v})'
                    # execution = self.execute_JOIN({v}, r)
                    initial_programs.append(Program(source=v,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function='JOIN',
                                                    derivations={v: [':' + r[:-4] if r[-4:] == '_inv' else '^:' + r]},
                                                    height=0,
                                                    finalized=False if self._dataset != 'webq' else True,
                                                    execution=(self.execute_JOIN, {v}, r)))
            else:
                if self._dataset == 'webq':
                    if len(v) <= 2 and re.match('[\d]{1}', v):
                        code = f'(JOIN sports.sports_team_roster.number {v})'
                        code_raw = f'(JOIN sports.sports_team_roster.number {v})'

                        initial_programs.append(Program(source=v,
                                                        code=code,
                                                        code_raw=code_raw,
                                                        function='JOIN',
                                                        derivations={v: ['^:sports.sports_team_roster.number']},
                                                        height=0,
                                                        finalized=False,
                                                        execution=(self.execute_JOIN, v, r)))
                    continue
                possible_attributes = self.get_attributes_for_value(v, use_ontology=False)
                # possible_attributes = random.sample(possible_attributes, 20)
                for r in possible_attributes:
                    code = f'(JOIN {r} {entity_name[v]})'
                    code_raw = f'(JOIN {r} {v})'
                    # execution = self.execute_JOIN(v, r)
                    initial_programs.append(Program(source=v,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function='JOIN',
                                                    derivations={v: ['^:' + r]},
                                                    height=0,
                                                    finalized=False if self._dataset != 'webq' else True,
                                                    execution=(self.execute_JOIN, v, r)))

                if self._dataset != "webq":
                    if not self._llm:
                        possible_attributes = self.get_attributes_for_value(v, use_ontology=True)
                    else:
                        possible_attributes = self.get_attributes_for_value(v, use_ontology=False)
                    for r in possible_attributes:
                        for comp in ["le", "ge", "lt", "gt"]:
                            code = f'({comp} {r} {entity_name[v]})'
                            code_raw = f'({comp} {r} {v})'
                            # execution = self.execute_Comparative(v, r, comp)
                            initial_programs.append(Program(source=v,
                                                            code=code,
                                                            code_raw=code_raw,
                                                            function=comp,
                                                            derivations={v: (['^:' + r], comp)},
                                                            height=0,
                                                            execution=(self.execute_Comparative, v, r, comp)))

        # The following is for (ARGXXX Class_Name Relation/Attribute)
        if self._dataset == 'webq':
            answer_types = []
        for at in answer_types:
            if self._llm and len(entity_name) > 0:
                break
            if at in self.cvt_types:
                continue
            possible_relations = self.get_relations_for_class(at, reverse=True)
            possible_relations.update(map(lambda x: x + '_inv',
                                          self.get_relations_for_class(at, reverse=False)))
            for r in possible_relations:
                for func in ['ARGMAX', 'ARGMIN']:
                    if r[-4:] == '_inv':
                        execution = self._relation_d[r[:-4]]
                    else:
                        execution = self._relation_r[r]
                    code = f'({func} {at} {r})'
                    code_raw = f'({func} {at} {r})'
                    initial_programs.append(Program(source=at,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function=func,
                                                    derivations={at: [':' + r[:-4] if r[-4:] == '_inv' else '^:' + r]},
                                                    height=0,
                                                    execution=execution))

            possible_attributes = self.get_attributes_for_class(at)
            for a in possible_attributes:
                for func in ['ARGMAX', 'ARGMIN']:
                    code = f'({func} {at} {a})'
                    code_raw = f'({func} {at} {a})'
                    initial_programs.append(Program(source=at,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function=func,
                                                    derivations={at: ['^:' + a]},
                                                    height=0,
                                                    # execution=(self.execute_JOIN, {v}, r),
                                                    finalized=True))
                if self.training and len(entity_name) == 0:
                    # this program makes no sense; only used to provide negative samples for superlatives
                    code = f'(JOIN {a} {" ".join(at.split(".")[-1].split("_"))})'
                    code_raw = code
                    initial_programs.append(Program(source=at,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function="JOIN",
                                                    derivations={at: ['^:' + a]},
                                                    height=0,
                                                    dummy=True))
                    code = f'(AND {at} (JOIN {a} {" ".join(at.split(".")[-1].split("_"))}))'
                    code_raw = code
                    initial_programs.append(Program(source=at,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function="AND",
                                                    derivations={at: ['^:' + a]},
                                                    height=0,
                                                    dummy=True))

        return initial_programs


if __name__ == '__main__':
    computer = Computer(dataset="webq")
