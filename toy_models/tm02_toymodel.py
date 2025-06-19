# general
import pandas as pd
import numpy as np
import datetime
import os

# custom func
from tm01_preprocess01 import load_pd_df_data, add_pn_doc, load_gensim_model, sentence2vec, add_qpn_vec, add_pn_idx_vec

# ml
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from datasets import Dataset as HFDataset, DatasetDict

# visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():

    #################################################################################################

    ### 
    ### load or preprocess data
    ### 

    #################################################################################################

    push_to_hf_repo = False
    saved_data_path = "tmp/train_data.pkl"

    if os.path.exists(saved_data_path):
        # this takes less than 10s
        df_train, df_test, df_val = load_pd_df_data()

        # total around 1m30s with ver1 data
        df_train = add_pn_doc(df_train)
        df_test = add_pn_doc(df_test)
        df_val = add_pn_doc(df_val)

        # this takes less than 30s
        word2vec = load_gensim_model('glove-twitter-25')

        # total around 11m30s
        df_train = add_qpn_vec(df_train, word2vec)
        df_test = add_qpn_vec(df_test, word2vec)
        df_val = add_qpn_vec(df_val, word2vec)

        # total les than 1m30s
        df_train = add_pn_idx_vec(df_train, word2vec)
        df_test = add_pn_idx_vec(df_test, word2vec)
        df_val = add_pn_idx_vec(df_val, word2vec)

        df_train.to_pickle("tmp/train_data.pkl")  # Or use .csv if you prefer
        df_test.to_pickle("tmp/test_data.pkl")  # Or use .csv if you prefer
        df_val.to_pickle("tmp/val_data.pkl")  # Or use .csv if you prefer

        if push_to_hf_repo:

            from huggingface_hub import login
            login()  # will prompt you for your token

            # save to huggingface dataset repo
            dataset_dict = DatasetDict({
                "train": HFDataset.from_pandas(df_train),
                "test": HFDataset.from_pandas(df_test),
                "val": HFDataset.from_pandas(df_val),
            })

            # Push to your Hugging Face account (repo will be created if it doesn't exist)
            dataset_dict.push_to_hub("hiki-t/marco_data_with_vecs")
    else:
        # once saved
        df_train = pd.read_pickle("tmp/train_data.pkl")
        df_test = pd.read_pickle("tmp/test_data.pkl")
        df_val = pd.read_pickle("tmp/val_data.pkl")

    #################################################################################################

    ### 
    ### prepare setting for training
    ### 

    #################################################################################################

    # load pre-trained word2vec model
    word2vec = load_gensim_model('glove-twitter-25')

    class ProjectionModel(nn.Module):
        def __init__(self, input_dim=25, projection_dim=25):
            super().__init__()
            self.project = nn.Sequential(
                nn.Linear(input_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )

        def forward(self, x):
            projected = self.project(x)
            return F.normalize(projected, p=2, dim=-1)  # L2 normalize for cosine similarity

    class VecTripletDataset(Dataset):
        def __init__(self, dataframe):
            self.df = dataframe

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            index_col = row["index"] # .to_numpy()
            vec_qry = np.stack(row["vec_query"].to_numpy())
            vec_posdoc = np.stack(row["vec_posdoc"].to_numpy())
            vec_negdoc = np.stack(row["vec_negdoc"].to_numpy())
            return {
                'row_index': torch.tensor(idx, dtype=torch.long), # this gives actual DataFrame row index
                'query': torch.tensor(vec_qry, dtype=torch.float32),
                'pos': torch.tensor(vec_posdoc, dtype=torch.float32),
                'neg': torch.tensor(vec_negdoc, dtype=torch.float32),
            }

    def info_nce_loss(query, pos_indices, all_or_many_vec_posdoc, temperature=0.07):
        # query: (B, D), pos: (B, D)
        # all_or_many_vec_posdoc: (N, D), full pool

        # Cosine similarities
        logits = torch.matmul(query, all_or_many_vec_posdoc.T)  # (B, N)

        # Scale by temperature
        logits /= temperature

        # Labels are the indices of the true positives
        labels = pos_indices.to(query.device)  # Make sure it's on same device

        # Use cross entropy: each query should match only its true doc
        return torch.nn.functional.cross_entropy(logits, labels)

    #################################################################################################

    ### 
    ### training, validation, playing
    ### 

    #################################################################################################

    #################### 

    ### parameter settings

    b_size = 64
    lr = 1e-3 # [0.01, 1e-3]
    num_epochs = 1
    topk = 1000 # num of top similar against true pos
    vis_freq = 3 # means every 3 epoch, visualization
    save_trained_model = False
    save_model_path = 'trained_models/qpn2vec_step1_repo/model_state.pth' # Path to saved model
    input_dim = 25 # as using twitter-25
    model_proj_dim = 25 # this is just inner projection dim

    #################### 

    # do this as there might be None values
    select_cols = ["query", "posDoc", "vec_query", "vec_posdoc", "vec_negdoc"]
    df_train_clean = df_train[df_train[["vec_query", "vec_posdoc", "vec_negdoc"]].notnull().all(axis=1)][select_cols]
    df_test_clean = df_test[df_test[["vec_query", "vec_posdoc", "vec_negdoc"]].notnull().all(axis=1)][select_cols]
    df_val_clean = df_val[df_val[["vec_query", "vec_posdoc", "vec_negdoc"]].notnull().all(axis=1)][select_cols]
    df_train_clean = df_train_clean.reset_index()
    df_test_clean = df_test_clean.reset_index()
    df_val_clean = df_val_clean.reset_index()

    # this is for evaluation
    all_vec_posdoc_np = np.stack(df_train_clean["vec_posdoc"].values)  # shape: (82324, 25) # Convert Series of arrays into a 2D numpy array
    all_vec_posdoc = torch.tensor(all_vec_posdoc_np, dtype=torch.float32)  # shape: (82324, 25) # Convert to torch tensor
    all_vec_posdoc_norm = torch.nn.functional.normalize(all_vec_posdoc, p=2, dim=1)  # shape: (82324, 25) # Normalize all vectors to unit norm (along dim=1, row-wise)
    all_vec_posdoc_norm.requires_grad_(False)  # <- disables gradient tracking
    POSDOC_POOL_SIZE = all_vec_posdoc_norm.size(0)  # should be 82324

    # Dataset & Dataloader
    dataset_train = VecTripletDataset(df_train_clean)
    train_loader = DataLoader(dataset_train, batch_size=b_size, shuffle=True)

    dataset_test = VecTripletDataset(df_test_clean)
    test_loader = DataLoader(dataset_test, batch_size=b_size, shuffle=True)

    dataset_val = VecTripletDataset(df_val_clean)
    val_loader = DataLoader(dataset_val, batch_size=b_size, shuffle=True)

    # Model instantiation
    model1 = ProjectionModel(input_dim=input_dim, projection_dim=model_proj_dim)
    optimizer = torch.optim.Adam(model1.parameters(), lr=lr)

    # Load trained weights if exists
    if os.path.exists(save_model_path):
        model1.load_state_dict(torch.load(save_model_path))
        print("Model loaded successfully from:", save_model_path)
    else:
        print("Model file not found at:", save_model_path)

    ###
    ### training starts here
    ###

    for epoch in range(num_epochs):
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Get current time and format it

        # Train 1
        total_train_loss = 0
        model1.train()
        for batch in train_loader:
            q = model1(batch['query'])  # shape (B, D)
            p = model1(batch['pos'])
            n = model1(batch['neg'])

            loss = info_nce_loss(q, batch['row_index'], all_vec_posdoc_norm, temperature=0.07)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"T1: Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

        # Visualization
        model1.eval()
        with torch.no_grad():
            if (epoch==0) or ((epoch+1) % vis_freq == 0):
                model1.eval()
                all_vecs = []
                all_labels = []  # "query", "pos", "neg"

                with torch.no_grad():
                    for batch in train_loader:
                        q = model1(batch['query'])  # (B, D)
                        p = model1(batch['pos'])    # (B, D)
                        n = model1(batch['neg'])    # (B, 10, D)

                        # Flatten n: (B, 10, D) → (B*10, D)
                        n_flat = n.view(-1, n.shape[-1])

                        all_vecs.append(q.cpu())
                        all_vecs.append(p.cpu())
                        all_vecs.append(n_flat.cpu())

                        all_labels.extend(["query"] * q.shape[0])
                        all_labels.extend(["pos"] * p.shape[0])
                        all_labels.extend(["neg"] * n_flat.shape[0])

                        # Limit samples to ~2000 for t-SNE
                        if len(all_labels) > 2000:
                            break

                # Final tensors
                vecs = torch.cat(all_vecs, dim=0).numpy()
                labels = np.array(all_labels)

                # t-SNE projection
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                vecs_2d = tsne.fit_transform(vecs)

                # Plot
                plt.figure(figsize=(10, 7))
                for label in ['query', 'pos', 'neg']:
                    idx = labels == label
                    plt.scatter(vecs_2d[idx, 0], vecs_2d[idx, 1], label=label, alpha=0.6, s=15)

                plt.title(f"Epoch{epoch+1}: t-SNE projection of Query / Positive / Negative Embeddings")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"results/vis/vis01_{timestamp}_epoch{epoch+1}_T1.png")
                # plt.show()
                plt.close()

        total = 0
        correct = 0

    if save_trained_model:
        torch.save(model1.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    # Evaluation: accuracy
    model1.eval()
    with torch.no_grad():
        # trained data
        for batch in train_loader:
            q_vecs = model1(batch['query'])  # (B, D)
            pos_indices = batch['row_index']  # Indices of correct positive docs in `all_doc_embeddings` (B,)

            # Compute cosine similarity between each query and all docs
            sims = torch.matmul(q_vecs, all_vec_posdoc_norm.T)  # (B, N)

            # Get Top-K most similar docs for each query
            topk_indices = torch.topk(sims, k=topk, dim=1).indices  # (B, topk)

            # Check how many times the true pos doc is in topk
            match = (topk_indices == pos_indices.unsqueeze(1)).any(dim=1)  # (B,)
            correct += match.sum().item()
            total += q_vecs.size(0)

        accuracy = correct / total
        print(f"Trained data, Top-{topk} Accuracy: {accuracy:.4f} ({correct} / {total})")

        # val data
        for batch in val_loader:
            q_vecs = model1(batch['query'])  # (B, D)
            pos_indices = batch['row_index']  # Indices of correct positive docs in `all_doc_embeddings` (B,)

            # Compute cosine similarity between each query and all docs
            sims = torch.matmul(q_vecs, all_vec_posdoc_norm.T)  # (B, N)

            # Get Top-K most similar docs for each query
            topk_indices = torch.topk(sims, k=topk, dim=1).indices  # (B, topk)

            # Check how many times the true pos doc is in topk
            match = (topk_indices == pos_indices.unsqueeze(1)).any(dim=1)  # (B,)
            correct += match.sum().item()
            total += q_vecs.size(0)

        accuracy = correct / total
        print(f"Val data, Top-{topk} Accuracy: {accuracy:.4f} ({correct} / {total})")

    # Evaluation: check what kind of similar items are picked for a query
    model1.eval()
    with torch.no_grad():
        for batch in train_loader:
            row_idx = batch['row_index'][0].item()
            print(f"the query for row index {row_idx} is `{df_train_clean['query'][row_idx]}`")
            print(f"1/posdoc_len for row index {row_idx} is `{df_train_clean['posDoc'][row_idx][0][:50]} ...`")
            q = model1(batch['query'])

            # Ensure q is shaped properly
            q = q[0].unsqueeze(0)  # shape: (1, 25)
            # Compute cosine similarity with all docs
            sims = cosine_similarity(q, all_vec_posdoc_norm)  # shape: (82324, 25)
            topk_values, topk_indices = torch.topk(sims, k=10)  # top 5

            for i, (score, idx) in enumerate(zip(topk_values, topk_indices)):
                print(f"Rank {i+1} | Score: {score:.4f}")
                print(f"Text: {df_train_clean['posDoc'][idx.item()]}")
                print("---")
            break

    # more wild test
    model1.eval()
    with torch.no_grad():
        sentence = "Where is the UK?"
        query_vec = torch.tensor(sentence2vec(word2vec, sentence), dtype=torch.float32)
        q = model1(query_vec)
        sims = cosine_similarity(q, all_vec_posdoc_norm)  # shape: (82324, 25)
        topk_values, topk_indices = torch.topk(sims, k=10)  # top 5

        for i, (score, idx) in enumerate(zip(topk_values, topk_indices)):
            print(f"Rank {i+1} | Score: {score:.4f}")
            print(f"Text: {df_train_clean['posDoc'][idx.item()]}")
            print("---")

if __name__ == "__main__":
    main()  # or whatever your main function is