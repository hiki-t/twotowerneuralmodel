import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, triplet_margin_loss
from torch.utils.data import DataLoader

# # load data
# fea = torch.load("data/features.pt")
# tar = torch.load("data/target.pt")
   
# # Hyperparameters
# learning_rate = 0.05
# batch_size = 64
# num_epochs = 20
# num_features = fea.shape[1] 

emb_dim = 25
hidden_dim = 10

# model
class TTSearch0(nn.Module):
    def __init__(self,  emb_dims, hidden_dim): 
       super().__init__()
       self.linear_left = nn.Linear(emb_dims, hidden_dim)
       self.linear_right = nn.Linear(emb_dims, hidden_dim)
       
    def forward(self, queries, pdocs, ndocs):
        ql = self.linear_left(queries)
        pl = self.linear_right(pdocs)
        nl = self.linear_right(ndocs)
        return (ql, pl, nl)
        
model = TTSearch0(emb_dim, hidden_dim)

criterion = nn.TripletMarginLoss()


# at training time
        ql, pl, nl = model
        loss = criterion(ql, pl, nl)
        loss.backward()
        optimizer.step()