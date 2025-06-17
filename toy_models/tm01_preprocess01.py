from datasets import load_dataset
import gensim.downloader as api
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # for visualization

### Step1. load datasets
def load_pd_df_data(v_num=0):
    dataset_versions = ['v1.1', 'v2.1']
    ds_ver = dataset_versions[v_num]
    dataset = load_dataset("microsoft/ms_marco", ds_ver)
    train_ds = dataset["train"]
    passages = train_ds["passages"]
    queries = train_ds["query"]

    df = pd.DataFrame([
        {
            "selected_index": row["is_selected"],
            "texts": row["passage_text"],
            "urls": row["url"],
            "query": queries[i],
        }
        for i, row in enumerate(passages)
    ])
    return df

# ### checking distribution of each row sentence size
# len_index = df["selected_index"].apply(len)
# print(len_index.min())
# print(len_index.max())
# plt.hist(len_index)

### Step2. load a trained model with specific datasets

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

# ### for practice

# print(pretrained_vec.get_vector("cat")) # get a trained vec

# probe_qry1 = "cat"
# test_pairs = [
#     [probe_qry1, "dog"], 
#     [probe_qry1, "monkey"], 
#     [probe_qry1, "bear"], 
#     [probe_qry1, "pet"], 
#     [probe_qry1, "girl"], 
# ]

# # get cosine similarity (-1.0 to 1.0)
# for row in test_pairs:
#     qry1 = pretrained_vec.get_vector(row[0])
#     qry2 = pretrained_vec.get_vector(row[1])
#     # Compute cosine similarity
#     cos_sim = np.dot(qry1, qry2) / (np.linalg.norm(qry1) * np.linalg.norm(qry2))
#     print(cos_sim)

### Step3. convert a sentence(words) to embedding vectors, average vectors to a vector

def sentence2vec(model, sentence):
    # lower characters
    words = sentence.lower().split()

    # Filter for known words only
    valid_vectors = [model.get_vector(word) for word in words if word in model]

    # Compute average vector
    if valid_vectors:
        # print(f"Sentence vector shape: {valid_vectors.shape}")
        sentence_vector = np.mean(valid_vectors, axis=0)
        return sentence_vector
    else:
        return print("No known words in the sentence!")

### Step4. compare a query and a sentence

def compute_cos_sim(model, sentence, query):
    vec_sentence = sentence2vec(model, sentence)
    vec_query = sentence2vec(model, query)

    # Compute cosine similarity
    cos_sim = np.dot(vec_sentence, vec_query) / (np.linalg.norm(vec_sentence) * np.linalg.norm(vec_query))
    return cos_sim

def main():
    which_trained_model = 8
    pick_trained_model = trained_models[which_trained_model] # see above for more models
    model = load_gensim_model(pick_trained_model)

    test_sentence = ("Since 2007, the RBA's outstanding reputation has been affected by the 'Securency' ")
    test_query = ("in Sydney, New South Wales and at the Business Resumption Site.")
    cos_sim = compute_cos_sim(model, test_sentence, test_query)
    print(f"cosine similarity is {cos_sim}")
    return cos_sim

# go to api later on
if __name__ == "__main__":
    main()