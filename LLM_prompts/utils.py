import json
import re


def format_prompt(prompt_template, teaching_examples, input):
    formatted_prompt = "\n\n".join(prompt_template["template"])
    # Format teaching examples
    if teaching_examples:
        formatted_teaching_examples = []
        demonstration_template = prompt_template["demo"]
        for teaching_example in teaching_examples:
            formatted_teaching_example = "\n".join(demonstration_template["template"])
            for k, v in demonstration_template["slots"].items():
                value = teaching_example.get(k, v["default_value"])
                if "mapping" in v:
                    value = v["mapping"][value]
                if isinstance(value, list):
                    value = v.get("join_by").join(value)
                formatted_teaching_example = formatted_teaching_example.replace(
                    f"$${k}$$", value
                )
            formatted_teaching_examples.append(formatted_teaching_example)
        formatted_prompt = formatted_prompt.replace(
            "$$demo$$", "\n\n".join(formatted_teaching_examples)
        )
    else:
        formatted_prompt = formatted_prompt.replace("$$demo$$", "")

    # Format input
    if "input" in prompt_template:
        input_template = prompt_template["input"]
        formatted_input = "\n".join(input_template["template"])
        for k, v in input.items():
            if isinstance(v, list):
                v = input_template["slots"][k].get("join_by").join(v)
            formatted_input = formatted_input.replace(f"$${k}$$", str(v))
        formatted_prompt = formatted_prompt.replace(f"$$input$$", formatted_input)
    else:  # Simplified template, just replace $$input$$
        formatted_prompt = formatted_prompt.replace(f"$$input$$", input["input"])
    return formatted_prompt.lstrip()
