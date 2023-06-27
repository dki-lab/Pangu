import torch
from utils.bert_interface import PretrainedTransformerEmbedder
from pytorch_transformers import BertConfig,BertTokenizer
model_type = "bert-base-uncased"
model = PretrainedTransformerEmbedder(model_name=model_type).cuda()
model = model.transformer_model
tokenizer = BertTokenizer.from_pretrained(model_type)

text1 = 'We met today and she wanted to'
text2 = 'meet again'
tok1 = tokenizer.tokenize(text1)
tok2 = tokenizer.tokenize(text2)

p_pos = len(tok1) # position for token
tok = tok1+tok2
tok,p_pos, tok[p_pos]

ids = torch.tensor(tokenizer.convert_tokens_to_ids(tok)).unsqueeze(0).to('cuda')
with torch.no_grad():
    output = model(ids)
attentions = torch.cat(output[3]).to('cpu')
print(attentions.shape) #(layer, batch_size (squeezed by torch.cat), num_heads, sequence_length, sequence_length)


attentions = attentions.permute(2,1,0,3)
print(attentions.shape) #(sequence_length, num_heads, layer, sequence_length)

layers = len(attentions[0][0])
heads = len(attentions[0])
seqlen = len(attentions)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
cols = 2
rows = int(heads/cols)

attentions_pos = attentions[p_pos]
print(attentions_pos.shape)

fig, axes = plt.subplots( rows,cols, figsize = (14,30))
axes = axes.flat
print (f'Attention weights for token {tok[p_pos]}')
for i,att in enumerate(attentions_pos):

    #im = axes[i].imshow(att, cmap='gray')
    sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok)
    axes[i].set_title(f'head - {i} ' )
    axes[i].set_ylabel('layers')


