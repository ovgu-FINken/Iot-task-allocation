import torch
import glob
import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

def drop_dim(x):
    x.y = torch.unsqueeze(x.y[0],0)
    return x    



class NS3DataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        print(glob.glob("trainingdata/data*.pt"))
        return glob.glob("trainingdata/data*.pt")

    @property
    def processed_file_names(self):
        return['processed_data.pt']


    def process(self):
        data_list = [drop_dim(torch.load(f)) for f in glob.glob("trainingdata/data*.pt")]
        data, slices = self.collate(data_list)
        data.y = F.normalize(data.y, dim=0)
        print(data.y)   
        #loader = DataLoader(data_list, batch_size = len(data_list))   
        #d = next(iter(loader))
        #means = [d.y[:,0].mean(), d.y[:,1].mean(), d.y[:,2].mean(), d.y[:,3].mean(), d.y[:,4].mean(), d.y[:,5].mean()]
        #stds =  [d.y[:,0].std(), d.y[:,1].std(), d.y[:,2].std(), d.y[:,3].std(), d.y[:,4].std(), d.y[:,5].std()])
        torch.save((data,slices), self.processed_paths[0])


if __name__ == "__main__":
    print("creating dataset..")
    ds = NS3DataSet(".")
    print(len(ds))
    print(ds.num_classes)
    print(ds.num_node_features)
    
    #data_list = [torch.load(f) for f in glob.glob("trainingdata/data*.pt")]
    #loader = DataLoader(data_list, batch_size=32)
    
    #for data in loader:
    #    print(data)
    #print()
    #for data in loader2:
    #    print(data)

    from torch_geometric.nn import GCNConv, DenseGCNConv, SAGEConv, global_mean_pool
    



    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            #self.conv1 = GCNConv(ds.num_node_features, 256)
            #self.conv2 = GCNConv(256, 256)
            #self.conv3 = GCNConv(256, 256)
            #self.conv4 = GCNConv(256, 256)
            #self.conv5 = GCNConv(256, 256)
            #self.conv6 = GCNConv(256, 256)
            #self.conv7 = GCNConv(256, 256)
            #self.dense1 = DenseGCNConv(256,ds.num_classes)
            self.conv1 = SAGEConv(-1,256)
            self.conv2 = SAGEConv(-1,256)
            self.conv3 = SAGEConv(-1,512)
            self.conv4 = SAGEConv(-1,512)
            self.conv5 = SAGEConv(-1,256)
            self.conv6 = SAGEConv(-1,256)
            self.lin1 = torch.nn.Linear(256, 6)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, edge_index)
            x = F.leaky_relu(x)
            x = self.conv3(x, edge_index)
            x = F.leaky_relu(x)
            x = self.conv4(x, edge_index)
            x = F.leaky_relu(x)
            x = self.conv5(x, edge_index)
            x = F.leaky_relu(x)
            x = self.conv6(x, edge_index)
            x = global_mean_pool(x, batch)
            x = F.dropout(x, training = self.training)
            x = self.lin1(x)

            return x
    
    loader = DataLoader(ds, batch_size=50, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #model=torch.load('/model.gnn')
    model = GCN()
    #model.load_state_dict(torch.load('/model_dict.pt'))
    model.to(device)
    
    def train_model():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=5e-4)
        #optimizer.load_state_dict(torch.load('/model_opt_dict.pt'))
        

        model.train()
        running_loss = 0.0
        for epoch in range(1000):
            for i,data in enumerate(loader): 
                d = data.to(device)
                optimizer.zero_grad()
                out = model(d)
                #out = out.reshape(d.y.size())
                loss = F.l1_loss(out, d.y.float())
                loss.backward()
                optimizer.step()
                

                running_loss += loss.item() 
                if i % 420 == 419:    # print every 2000 mini-batches
                    print(loss.item())
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/420:.6f}')
                    running_loss = 0.0
 
    train_model()
    model.eval()
    
    #torch.save(model, '/model.pt')
    torch.save(model.state_dict(), 'model_dict.pt')
    torch.save(optimizer.state_dict(), 'model_opt_dict.pt') 

    
    #print(ds[0].y)
    #pred = model(ds.to(device))
    #print(pred)
    
    
    #print(pred)

    #correct = (pred[0] == data.y).sum()
    #acc = int(correct) / int(data.y.sum())
    #print(f'Accuracy: {acc:.4f}')

