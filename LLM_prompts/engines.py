import os
import time

import backoff
import openai
import requests
from openai.error import APIError, RateLimitError, APIConnectionError, ServiceUnavailableError
# from transformers import BloomTokenizerFast, GPT2TokenizerFast, PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast


class Engine:
    def __init__(self) -> None:
        pass

    def tokenize(self, input):
        return self.tokenizer(input)


class OpenaiEngine(Engine):
    def __init__(
            self,
            api_key=None,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        Engine.__init__(self, **kwargs)

    @backoff.on_exception(backoff.expo, RateLimitError)
    def generate(self, prompts, max_new_tokens, stop=None, model=None, **kwargs):
        """
        See more in https://beta.openai.com/docs/api-reference/completions
        """
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        openai.api_key = self.api_keys[self.current_key_idx]
        if isinstance(prompts, str):
            prompts = [prompts]
        final_responses = [[] for i in range(len(prompts))]
        responses = openai.Completion.create(
            model=model if model else self.model,
            prompt=prompts,
            max_tokens=max_new_tokens,
            stop=stop if stop else self.stop,
            temperature=self.temperature,
            **kwargs
        )
        for choice in responses["choices"]:
            final_responses[choice["index"]].append(choice)
        # If success, then update the next avail time for the key
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
        return [
            {
                "raw": response,
                "outputs": [choice["text"] for choice in response],
            }
            for response in final_responses
        ]

    @backoff.on_exception(backoff.expo, RateLimitError)
    def chat_generate(self, messages, max_new_tokens, stop=None, model=None, **kwargs):
        """
        See more in https://beta.openai.com/docs/api-reference/completions
        """
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        openai.api_key = self.api_keys[self.current_key_idx]
        responses = openai.ChatCompletion.create(
            model=model if model else self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            stop=stop if stop else self.stop,
            temperature=self.temperature,
            **kwargs
        )
        # If success, then update the next avail time for the key
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
       
        return responses

    @backoff.on_exception(backoff.expo, (
    RateLimitError, openai.error.Timeout, APIError, APIConnectionError, ServiceUnavailableError,
    openai.error.InvalidRequestError))
    def score(self, prompts, targets, model=None, **kwargs):
        """
        See more in https://beta.openai.com/docs/api-reference/completions
        """
        # print(self.current_key_idx)
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        openai.api_key = self.api_keys[self.current_key_idx]
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        final_responses = [[] for i in range(len(prompts))]
        responses = openai.Completion.create(
            model=model if model else self.model,
            prompt=[prompt + target for prompt, target in zip(prompts, targets)],
            max_tokens=0,
            temperature=0,
            echo=True,
            logprobs=0,
            **kwargs
        )
        for choice in responses["choices"]:
            prompt_length = len(prompts[choice["index"]])
            target_offset = 0
            for i, offset in enumerate(choice["logprobs"]["text_offset"][::-1]):
                if offset < prompt_length:
                    target_offset = i
                    break
            final_responses[choice["index"]].append(
                {
                    "token_logprobs": choice["logprobs"]["token_logprobs"][-i:],
                    "tokens": choice["logprobs"]["tokens"][-i:],
                    "sequence_logprob": sum(
                        [
                            logprob
                            for logprob in choice["logprobs"]["token_logprobs"][-i:]
                            if logprob
                        ]
                    )
                                        / i,
                }
            )
        # If success, then update the next avail time for the key
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
        return [
            {
                "raw": response,
                "outputs": [choice["sequence_logprob"] for choice in response],
            }
            for response in final_responses
        ]


