import torch
import glob
import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


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
        data_list = [torch.load(f) for f in glob.glob("trainingdata/data*.pt")]
        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])


if __name__ == "__main__":
    ds = NS3DataSet(".")
    print(len(ds))
    print(ds.num_classes)
    print(ds.num_node_features)
    
    data_list = [torch.load(f) for f in glob.glob("trainingdata/data*.pt")]
    loader = DataLoader(data_list, batch_size=32)
    loader2 = DataLoader(ds, batch_size=32)
    
    for data in loader:
        print(data)
    print()
    for data in loader2:
        print(data)

    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(ds.num_node_features, 16)
            self.conv2 = GCNConv(16, ds.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = ds[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        print(out)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()


    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[0] == data.y).sum()
    acc = int(correct) / int(data.y.sum())
    print(f'Accuracy: {acc:.4f}')

