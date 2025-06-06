from typing import Dict

from overrides import overrides
from pathlib import Path

from transformers import AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler

import torch
from torch import nn

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

path = str(Path(__file__).parent.absolute())

# copied from AllenNLP 0.9.0, because the new one is incompatible with my implementation
def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor],
                        num_wrapping_dims: int = 0) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.

    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.

    TODO(joelgrus): can we change this?
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors  makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.V(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    if "mask" in text_field_tensors:
        return text_field_tensors["mask"]

    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


class DistilBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.dim, config.dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@TokenEmbedder.register("huggingface_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``Transformers`` as a ``TokenEmbedder``.
    """

    def __init__(self, model_name: str, layers_to_freeze=None, pooling=False) -> None:
        super().__init__()
        self.model_name = model_name
        # todo: this part is too ad-hoc and error-prone; replace it with Huggingface's official guideline
        if "roberta" in model_name or "code" in model_name or "deberta" in model_name:
            self.EOS = 2
        elif "bert" in model_name:  # works for bert and distilbert
            self.EOS = 102
        # config = PretrainedConfig.from_json_file(path + "/../bert_configs/debug.json")
        # self.transformer_model = AutoModel.from_pretrained(model_name, config=config)
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

        if "deberta" in model_name:
            self.context_pooler = ContextPooler(self.transformer_model.config)
        elif "distilbert" in model_name:
            self.context_pooler = DistilBertPooler(self.transformer_model.config)

        if layers_to_freeze is not None:
            modules = [self.transformer_model.embeddings,
                       *self.transformer_model.encoder.layer[:layers_to_freeze]]  # Replace 5 by what you want
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self._pooling = pooling

        # attentions, ylabels, xlabels = self.attention_viz("Which wine in Tulum valley has the most alcohol?",
        #                                                   [
        #                                                    "common.topic",
        #                                                    "business.consumer_product", "wine.wine"])

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore
        if "roberta" in self.model_name or "code" in self.model_name:
            # In RoBerta's vocabulary, padding is not mapped to 0 as allennlp's default setting,
            # instead, it is mapped to 1, so here we temporarily replace 1 to 0, while 0 stands for <s>,
            # we just replace 0 to a random non-padding id, e.g., 23 here
            attention_mask = get_text_field_mask({'bert': token_ids.masked_fill(token_ids == 0, 23)
                                                 .masked_fill(token_ids == 1, 0)})
            # roberta doesn't have type ids, it distinguish the first and second sentence only based on </s>
            token_type_ids = None
        elif "bert" in self.model_name:   # works for bert, deberta, and distilbert
            attention_mask = get_text_field_mask({'bert': token_ids})
            if "distilbert" not in self.model_name:  # distilbert has no token type ids
                token_type_ids = self.get_type_ids(token_ids)

        # attention_mask = None
        # pylint: disable=arguments-differ
        # position_ids = self.get_position_ids(token_ids).to(token_ids.device)
        # token_type_ids = None
        position_ids = None

        if "distilbert" not in self.model_name:
            outputs = self.transformer_model(token_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                                             attention_mask=attention_mask)
        else:
            outputs = self.transformer_model(token_ids, attention_mask=attention_mask)

        if "deberta" not in self.model_name and "distilbert" not in self.model_name:
            if self._pooling:  # this works for BERT and RoBERTa
                return outputs[1]
            else:
                return outputs[0]
        else:  # for deberta and distilbert
            if self._pooling:
                return self.context_pooler(outputs[0])
            else:
                return outputs[0]

    def get_type_ids(self, token_ids: torch.LongTensor):
        type_ids = torch.zeros_like(token_ids)
        num_seq, max_len = token_ids.shape
        for i in range(num_seq):
            for j in range(max_len):
                if token_ids[i][j] == self.EOS:  # id of [SEP] or </s>'s first occurence
                    break
            type_ids[i][j + 1:] = 1
        return type_ids