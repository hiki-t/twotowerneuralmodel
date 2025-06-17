# from  torchtext.data.utils import get_tokenizer
import gensim.downloader as api
from gensim.utils import simple_preprocess
import pandas as pd
import torch

wv = api.load('glove-twitter-25')

# read data
qs = pd.read_parquet('temp/queries.parquet.gzip')
docs = pd.read_parquet('temp/passages.parquet.gzip')
triplets = pd.read_parquet('temp/triplets.parquet.gzip') 

#tokenize and average-pool
def process(text):
    t = simple_preprocess(text)
    e = []
    for w in t:
        try:
            e.append(torch.tensor(wv[w]))
        except KeyError:
            continue
    if len(e)<1:
        return torch.zeros(25)
    else:
        return torch.mean(torch.stack(e), axis=0)

qs_emb = torch.stack([process(s) for s in qs['query']])
docs_emb = torch.stack([process(s) for s in docs['passage_text']])
tri = torch.tensor(triplets.values)

torch.save(qs_emb, 'temp/qs_tensor.pt')
torch.save(docs_emb, 'temp/docs_tensor.pt')
torch.save(tri, 'temp/tri_tensor.pt')


