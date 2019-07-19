
import torch
import torch.nn as nn
import torch.nn.functional as F

from fuzzy_relu import fuzzy_relu

class DagSage(nn.Module):
    """
    A graphsage inspired convolution on a directional acyclic graph
    """
    def __init__(self, input_dim, output_dim, representation_size):
        # input_dim: size of vector representation of incoming nodes
        # output_dim: size of node output dimension per node
        # representation_size: size of internal hidden layers
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_size = representation_size
        self.node_self_rep = nn.Linear(input_dim, representation_size)
        self.src_representation = nn.Linear(output_dim, representation_size)

    def cuda(self):
        self.node_self_rep = self.node_self_rep.cuda()
        self.src_representation = self.src_representation.cuda()
        return self

    def forward(self, nodes_adj):
        # nodes_adj[0]: (batch, node, vector) - node representations
        # nodes_adj[1]: (batch, node^2)
        # nodes_adj[1] is a directional adjacency matrix, can accept non-binary inputs

        node_id_rep = self.node_self_rep(nodes_adj[0])


        src_representation = self.src_representation(nodes_adj[0])
        # Normalize inputs to each node
        conn = F.normalize(nodes_adj[1], dim=2)




