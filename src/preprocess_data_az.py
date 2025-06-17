import pandas as pd
import numpy as np

random.seed(12345)
np.random.seed(67890)
batch_size_for_sampling_negs = 500


# main dataset
raw_train = pd.read_parquet("data/train-00000-of-00001.parquet")
raw_test = pd.read_parquet("data/test-00000-of-00001.parquet")
raw_valid = pd.read_parquet("data/validation-00000-of-00001.parquet")

# combine and unfold
raw_ds = pd.concat([raw_train, raw_test, raw_valid], ignore_index=True)[["query_id","query","passages"]]

#  inflate the dataset
ds_long = raw_ds.drop(columns = 'passages').join(pd.json_normalize(raw_ds["passages"])).explode(column = ['is_selected', 'passage_text', 'url'])
 
# create a table with passages
ds_long['pass_id'] = ds_long.groupby(['passage_text','url'], sort=False).ngroup()
uni_passages = ds_long.drop_duplicates(['pass_id','passage_text','url'])[['pass_id','passage_text','url']]

# create a table with queries
ds_long['q_id'] = ds_long.groupby(['query_id'], sort=False).ngroup()
uni_queries = ds_long.drop_duplicates(['q_id','query_id','query'])[['q_id','query_id','query']]

## make a triplet -- randomly pick a comparison passage
all_pass_ids = uni_passages['pass_id'].unique()
def sample_negatives(group):
    pos_ids = group['pass_id'].unique()
    available_neg = np.setdiff1d(all_pass_ids, pos_ids, assume_unique=True)
    return pd.Series(np.random.choice(available_neg, size=len(group), replace=False), index=group.index)

ds_long['pos'] = ds_long['pass_id']
ds_long['batch'] = ds_long['q_id'] // batch_size_for_sampling_negs
ds_long['neg'] = ds_long.groupby('batch', group_keys=False).apply(sample_negatives)

# save datasets
uni_passages.to_parquet('temp/passages.parquet.gzip',
              compression='gzip')
uni_queries.to_parquet('temp/queries.parquet.gzip',
              compression='gzip')
ds_long[['q_id','pos','neg']].to_parquet('temp/triplets.parquet.gzip',
              compression='gzip') 













