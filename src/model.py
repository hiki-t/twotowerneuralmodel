import torch
import torch.nn as nn 
from torch.nn.functional import cosine_similarity, triplet_margin_loss
from torch.utils.data import DataLoader 

torch.backends.cudnn.deterministic = True 
torch.manual_seed(1234)
torch.cuda.manual_seed_all(5678) 

## load data
qs = torch.load('temp/qs_tensor.pt')
docs = torch.load('temp/docs_tensor.pt')
tri = torch.load('temp/tri_tensor.pt')

## train test split 
sel  = torch.rand(len(tri))
train_dataset = tri[sel<=0.99]
test_dataset = tri[sel>0.99]

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

def make(config):
    # Make the data 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
 
    # Make the model 
    model = TTSearch0(config['emb_dim'], config['hidden_dim']).to(config['device'])

    # Make the loss and optimizer
    criterion = nn.TripletMarginLoss(margin = config['margin'], swap=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, config): 
    model.train()
    for epoch in range(config['num_epochs']):
        losses = []
        for t in train_loader: 
            qe = qs[t[:,0]].to(config['device'])
            pe = docs[t[:,1]].to(config['device'])
            ne = docs[t[:,2]].to(config['device'])
            
            optimizer.zero_grad()
            qv, pv, nv = model(qe, pe, ne)
            loss = criterion(qv, pv, nv)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        mean_loss = torch.stack(losses).mean()
        print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}')

def test(model, test_data):
    model.eval() 
    qe = qs[test_data[:,0]].to(config['device'])
    pe = docs.to(config['device'])
    ne = docs[:2].to(config['device'])
    qv, pv, _ = model(qe,pe,ne)
    pv = pv / torch.linalg.vector_norm(pv, dim=1).unsqueeze(1)

    cossims_this = []
    propbelow_this = [] 
    for i, u in enumerate(qv):
        v = torch.div(u, torch.linalg.vector_norm(u, dim=0)) 
        cossims = torch.matmul(pv, v).squeeze() 
        cossim_pos = cossims[test_data[i,1]] 
        cossims_this.append(cossim_pos)
        propbelow_this.append(torch.sum(cossims < cossim_pos)/len(cossims)) 

    ave_cossim = torch.cat([v.reshape(1) for v in cossims_this]).mean()
    ave_propbelow = torch.cat([v.reshape(1) for v in propbelow_this]).mean() 
    print(f'Average cosine similarity between the query and its true positive: {ave_cossim:.4f}') 
    print(f'Average share of documents with lower similarity to the query than the true positive: {ave_propbelow:.4f}') 

config = dict(
    num_epochs=10,
    batch_size=64,
    learning_rate=0.05,
    emb_dim = 25,
    hidden_dim = 20,
    margin = 0.5,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )


# execution
model, train_loader, criterion, optimizer = make(config) 
print(model) 
train(model, train_loader, criterion, optimizer, config)
test(model, test_dataset[:2500]) 
 









