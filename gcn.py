import torch
from torch_geometric.nn import GCNConv, DenseGCNConv, SAGEConv, global_mean_pool
import torch.nn.functional as F

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


if __name__ == "__main__":
  import sys
  model = GCN()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.load_state_dict(torch.load('model_dict.pt', map_location=device))
  model.to(device)
  print(sys.getsizeof(model))
  print(sys.getsizeof(model.state_dict()))
  s = 0
  for key,value in model.state_dict().items():
      s += sys.getsizeof(key)
      s += sys.getsizeof(value)
  print(s)
  mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
  mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
  mem = mem_params + mem_bufs # in bytes
  print(mem)


