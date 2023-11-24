import random
import re

from tqdm import tqdm
from collections import defaultdict
from utils.logic_form_util import same_logical_form, postprocess_raw_code
from utils.semparse_util import lisp_to_nested_expression, expression_to_lisp

from openai_eval.prompts import manual_prompt_short


def process_entities(expression):
    if isinstance(expression, list) and len(expression) > 3:
        expression[2] = '_'.join(expression[2:])
        expression = expression[:3]
    for i, e in enumerate(expression):
        if isinstance(e, list):
            expression[i] = process_entities(e)

    return expression


def templating_one_example(quesiton, program0, program1, label):
    """
    Template for one sample input
    :param quesiton: an input question
    :param program0: first candidate query
    :param program1: second candidate query
    :param label: which query matches the question better; empty for inference
    :return: a grounded template
    """
    # template 1
    template = f"Question: {quesiton}\nCandidate query A: {program0}\nCandidate query B: {program1}\n" \
               f"Which candidate matches the question intent better:{label}"

    return template


def create_prompt(in_context_samples, test_sample):
    prompt = ''
    for sample in in_context_samples:
        # sample[1] is always the positive one
        if random.random() > 0.5:
            prompt += templating_one_example(sample[0], sample[1], sample[2], 'A')
        else:
            prompt += templating_one_example(sample[0], sample[2], sample[1], 'B')
        prompt += '\n'
    prompt += templating_one_example(test_sample[0], test_sample[1], test_sample[2], '')

    return prompt


def manual_prompt(test_sample):
    prompt = manual_prompt_short + '\n'
    prompt = prompt + templating_one_example(test_sample[0], test_sample[1], test_sample[2], '')

    return prompt


def process_training_dump(file_name):
    processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: set())))
    with open(file_name) as f:
        for line in f:
            fields = line.split('\t')
            processed_data[fields[1]][int(fields[0])][int(fields[3])].add(fields[2])

    return processed_data


def random_samples(file_name, n=10):
    """
    Return a set of in-context samples in the format of (question, program_a, program_b), where program_a is positive
    while program_b is negative.
    :param file_name: file_name to the training data dump
    :param n: number of in-context samples expected
    :return: see the overall description
    """
    data_dump = process_training_dump(file_name)
    samples = []
    questions = set()
    while len(questions) < n:
        question_i = random.sample(list(data_dump.keys()), 1)[0]
        if question_i in questions:
            continue
        questions.add(question_i)
        all_items_i = data_dump[question_i]
        steps = len(all_items_i)
        for j in range(steps):
            golds = all_items_i[j][1]

            if len(all_items_i[j][0]) > 0:
                # randomly sample one
                negative0 = random.sample(all_items_i[j][0], 1)[0]

                for gold in golds:
                    # todo: same_logical_form is problematic with entity mentions (i.e., entity mentions have spaces)
                    # if same_logical_form(gold, negative0):
                    #     print(gold, negative0)
                    #     continue
                    # if random.random() > 0.5:
                    if random.random() > -100:
                        samples.append((question_i, gold, negative0))

            if j > 0:
                for neg in all_items_i[j][0]:
                    # used to learn the partial order of sub-programs
                    for gold in golds:
                        if gold.__contains__(neg):
                            # if neg != negative0 and random.random() > 0.5:
                            if neg != negative0:
                                samples.append((question_i, gold, neg))

    return samples


def is_valid(test_sample):
    if test_sample[1].__contains__("(l") or test_sample[1].__contains__("(g") \
            or test_sample[1].__contains__("(ARG") or test_sample[2].__contains__("(l") \
            or test_sample[2].__contains__("(g") or test_sample[2].__contains__("(ARG") \
            or test_sample[2].__contains__("(COUNT") or test_sample[1].__contains__("(COUNT"):
        return False

    try:
        expression1 = lisp_to_nested_expression(test_sample[1])
        expression2 = lisp_to_nested_expression(test_sample[2])
        expression1 = process_entities(expression1)
        expression2 = process_entities(expression2)
    except TypeError:
        return False

    lf1 = postprocess_raw_code(expression_to_lisp(expression1))
    lf2 = postprocess_raw_code(expression_to_lisp(expression2))
    if same_logical_form(lf1, lf2) and not (lf1.__contains__(lf2) or lf2.__contains__(lf1)):
        return False

    return True


