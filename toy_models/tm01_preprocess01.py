# general
import numpy as np
import pandas as pd
import random
from functools import partial

# ml
import gensim.downloader as api
from datasets import load_dataset

# visualization
# import matplotlib.pyplot as plt # for visualization


############################################################################################

### table of contents
# --- func: load datasets from huggingface
# df_train, df_test, df_val = load_pd_df_data(v_num=0)
# 
# ---func: add posDoc and negDoc on df
# df_data = add_pn_doc(df_data, select_k=10, rnd_seed=42)
# 
# --- func: load pretrained word2vec model
# word2vec = load_gensim_model(which_trained_model='glove-twitter-25')
# 
# --- func: convert a sentence(words) to embedding vectors, average the vectors to a vector
# None or sentence_vec = sentence2vec(word2vec, sentence)
# 
# --- func: func: convert sentence to qry, pos/neg doc embedding vectors, average the vectors to a vector
# None or df_data = add_qpn_vec(df_data, word2vec_model)
# 
# --- func: convert a sentence(words) within row with label 0s and a 1 to embedding vectors, average the vectors to a vector
# None or df_data = add_pn_idx_vec(df_data, word2vec_model, rnd_seed=42, target_len=2, rep_time=4)

############################################################################################


### func: load datasets from huggingface

def load_pd_df_data(v_num=0):
    def clean_data(pass_data, qry_data):
        clean_df_data = pd.DataFrame([
            {
                "selected_index": row["is_selected"],
                "texts": row["passage_text"],
                "urls": row["url"],
                "query": qry_data[i],
            }
            for i, row in enumerate(pass_data)
        ])
        return clean_df_data

    dataset_versions = ['v1.1', 'v2.1']
    ds_ver = dataset_versions[v_num]
    dataset = load_dataset("microsoft/ms_marco", ds_ver)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    pass_train = train_ds["passages"]
    qry_train = train_ds["query"]
    pass_test = test_ds["passages"]
    qry_test = test_ds["query"]
    pass_val = val_ds["passages"]
    qry_val = val_ds["query"]

    df_train = clean_data(pass_train, qry_train)
    df_test = clean_data(pass_test, qry_test)
    df_val = clean_data(pass_val, qry_val)

    return df_train, df_test, df_val

### func: add posDoc and negDoc on df

def add_pn_doc(df_data, select_k=10, rnd_seed=42):
    # Set random seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)

    select_k = select_k # decide how many negDoc to use
    df_data["posDoc"] = df_data["texts"] # add posDoc
    texts_list = df_data["texts"].tolist() # Convert df['texts'] to list for fast access
    num_rows = len(df_data)
    all_indices = np.arange(num_rows) # Precompute all_indices ONCE
    neg_docs = [None] * num_rows # Prepare output list

    for idx in range(num_rows):
        # Sample k other indices without removing idx from the array
        # Use set difference without np.delete (slow)
        while True:
            sampled_indices = np.random.choice(num_rows, size=select_k + 1, replace=False)
            if idx not in sampled_indices:
                sampled_indices = sampled_indices[:select_k]
                break

        # Collect texts
        sampled_texts = [texts_list[i] for i in sampled_indices]
        flattened_texts = [item for sublist in sampled_texts for item in sublist]
        neg_docs[idx] = flattened_texts

    # Assign column
    df_data["negDoc"] = neg_docs
    return df_data

### func: load a pre-trained word2vec model with specific datasets

# which_trained_model = 8
# print(list(gensim.downloader.info()['models'].keys())) # you can check trained datasets here
# all models can be found here: https://github.com/piskvorky/gensim-data
trained_models = ['fasttext-wiki-news-subwords-300',
'conceptnet-numberbatch-17-06-300',
'word2vec-ruscorpora-300',
'word2vec-google-news-300',
'glove-wiki-gigaword-50',
'glove-wiki-gigaword-100',
'glove-wiki-gigaword-200',
'glove-wiki-gigaword-300',
'glove-twitter-25',
'glove-twitter-50',
'glove-twitter-100',
'glove-twitter-200',
'__testing_word2vec-matrix-synopsis']

def load_gensim_model(which_trained_model='glove-twitter-25'):
    pretrained_model = api.load(which_trained_model)  # load glove vectors
    return pretrained_model

### func: convert a sentence(words) to embedding vectors, average the vectors to a vector

def sentence2vec(model, sentence):
    if not isinstance(sentence, str):
        return None  # Safely ignore non-string values like NaN

    words = sentence.lower().split() # lower characters
    valid_vectors = [model.get_vector(word) for word in words if word in model] # Filter for known words only

    # Compute average vector
    return np.mean(valid_vectors, axis=0) if valid_vectors else None

### func: convert sentence to qry, pos/neg doc embedding vectors, average the vectors to a vector

def add_qpn_vec(df_data, word2vec_model):

    num_rows = len(df_data)
    rows_posdoc = [None] * num_rows
    rows_negdoc = [None] * num_rows
    rows_qry = [None] * num_rows

    row_idx = 0
    for row in df_data.itertuples(index=False):
        qry = row.query
        pos_docs = row.posDoc
        neg_docs = row.negDoc

        # Vectorize posDoc
        vecs_posdoc = [sentence2vec(word2vec_model, s) for s in pos_docs]
        vecs_posdoc = [v for v in vecs_posdoc if v is not None]
        mean_vec_posdoc = np.mean(vecs_posdoc, axis=0) if vecs_posdoc else None

        # Vectorize negDoc
        vecs_negdoc = [sentence2vec(word2vec_model, s) for s in neg_docs]
        vecs_negdoc = [v for v in vecs_negdoc if v is not None]
        mean_vec_negdoc = np.mean(vecs_negdoc, axis=0) if vecs_negdoc else None

        # Vectorize query
        vec_qry = sentence2vec(word2vec_model, qry)
        vec_qry = vec_qry if vec_qry is not None else None

        rows_posdoc[row_idx] = mean_vec_posdoc
        rows_negdoc[row_idx] = mean_vec_negdoc
        rows_qry[row_idx] = vec_qry
        row_idx += 1

    # Assign vector columns
    df_data["vec_query"] = rows_qry
    df_data["vec_posdoc"] = rows_posdoc
    df_data["vec_negdoc"] = rows_negdoc
    return df_data

### func: convert a sentence(words) within row with label 0s and a 1 to embedding vectors, average the vectors to a vector

def sample_indices(index_list, target_len=2, rep_time=3):
    if len(index_list) < target_len+1:
        return None, None
    
    # Get indices for 0s and 1s
    neg_indices = [i for i, val in enumerate(index_list) if val == 0]
    pos_indices = [i for i, val in enumerate(index_list) if val == 1]
    
    # If not enough negative or positive labels, return None
    if len(neg_indices) < target_len or len(pos_indices) < 1:
        return None, None

    neg_samples = []
    pos_samples = []
    
    for _ in range(rep_time):
        neg_sample = random.sample(neg_indices, target_len)
        pos_sample = random.sample(pos_indices, 1)
        neg_samples.append(neg_sample)
        pos_samples.append(pos_sample)
    
    return neg_samples, pos_samples

def process_row(row, word2vec):
    texts = row["texts"]
    neg_idxs_list = row["neg_idxs"]
    pos_idxs_list = row["pos_idxs"]

    # If either is None (invalid case), return None
    if neg_idxs_list is None or pos_idxs_list is None:
        return None, None

    # Process negative indices: flatten list of lists (3 x 2 = 6 indices)
    neg_vecs = []
    for sublist in neg_idxs_list:
        for idx in sublist:
            sentence = texts[idx]
            vec = sentence2vec(word2vec, sentence)
            if vec is not None:
                neg_vecs.append(vec)

    # Average negative vectors
    if not neg_vecs:
        vec_neg_avg = None
    else:
        vec_neg_avg = np.mean(neg_vecs, axis=0)

    # Process positive index: take the first one only
    pos_idx = pos_idxs_list[0][0]
    sentence_pos = texts[pos_idx]
    vec_pos = sentence2vec(word2vec, sentence_pos)
    if vec_pos is None:
        return None, None

    return vec_neg_avg, vec_pos

def add_pn_idx_vec(df_data, word2vec_model, rnd_seed=42, target_len=2, rep_time=4):
    # Set random seed
    random.seed(rnd_seed); np.random.seed(rnd_seed); 
    results = df_data["selected_index"].apply(lambda x: sample_indices(x, target_len, rep_time)) # Apply function to each row
    df_data["neg_idxs"], df_data["pos_idxs"] = zip(*results) # Split the tuples into two new columns

    # Wrap the function with the model baked in
    process_row_with_model = partial(process_row, word2vec=word2vec_model)

    vecs = df_data.apply(process_row_with_model, axis=1) # Apply the function row-wise
    df_data["vec_neg_idxs"], df_data["vec_pos_idxs"] = zip(*vecs) # Unpack results into new columns
    return df_data

### func: compare a query and a sentence

def compute_cos_sim(model, sentence, query):
    vec_sentence = sentence2vec(model, sentence)
    vec_query = sentence2vec(model, query)

    # Compute cosine similarity
    cos_sim = np.dot(vec_sentence, vec_query) / (np.linalg.norm(vec_sentence) * np.linalg.norm(vec_query))
    return cos_sim

### StepN. all combined

def main():
    which_trained_model = 8
    pick_trained_model = trained_models[which_trained_model] # see above for more models
    model = load_gensim_model(pick_trained_model)

    test_sentence = ("Since 2007, the RBA's outstanding reputation has been affected by the 'Securency' ")
    test_query = ("in Sydney, New South Wales and at the Business Resumption Site.")
    cos_sim = compute_cos_sim(model, test_sentence, test_query)
    print(f"cosine similarity is {cos_sim}")
    return cos_sim

# # go to api later on
# if __name__ == "__main__":
#     main()