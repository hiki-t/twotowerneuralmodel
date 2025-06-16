import pandas as pd
import numpy as np
import random

# import torch
# import os
# from datetime import datetime
# import json

random.seed(12345)
np.random.seed(67890)

# main dataset
raw_train = pd.read_parquet("data/train-00000-of-00001.parquet")
raw_test = pd.read_parquet("data/test-00000-of-00001.parquet")

# combine and unfold
raw_train["test"] = False
raw_test["test"] = True
raw_ds = pd.concat((raw_train, raw_test))[["query_id","query","passages","test"]]

# count passages and inflate the dataset
maxlen = max(raw_ds['passages'].apply(lambda x: len(x.get("is_selected"))))

for i in range(maxlen):
    raw_ds[f'is_selected{i}'] = np.nan
    raw_ds[f'passage_text{i}'] = np.nan
    raw_ds[f'url{i}'] = np.nan

for j, r in raw_ds.iterrows():
    i = 0
    p = r['passages']
    for s, t, u in zip(p.get("is_selected"), p.get("passage_text"), p.get("url")):
      raw_ds.loc[j, f'is_selected{i}'] = s
      raw_ds.loc[j, f'passage_text{i}'] = t
      raw_ds.loc[j, f'url{i}'] = u 
      i += 1 
    
ds_long = pd.wide_to_long(
    raw_ds.drop(columns = "passages"),
    stubnames = ["is_selected", "passage_text","url"],
    i = "query_id",
    j = "pass_qid"
    ).reset_index().dropna()


uni_passages = ds_long.drop_duplicates(['passage_text','url'])[['passage_text','url']]
uni_passages = uni_passages.reset_index(drop=True).rename_axis(['pass_id']).reset_index()
# save the dictionary
uni_passages.to_parquet('temp/passages.parquet.gzip',
              compression='gzip')

ds_long_2 = pd.merge(ds_long, uni_passages, on = ['passage_text','url'])

## make a triplet -- randomly pick a comparison passage
ds_long_2['pos'] = ds_long_2['pass_id']
ds_long_2['neg'] = ds_long_2.groupby(by='query_id')['pass_id'].transform(
    lambda vec: random.sample(sorted(set(uni_passages['pass_id']).difference(set(vec))), len(vec))
    )

## save the master dataset
ds_long_2[['query_id','query','pos','neg','test']].to_parquet('temp/master.parquet.gzip',
              compression='gzip') 













