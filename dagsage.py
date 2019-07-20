
import torch
import torch.nn as nn
import torch.nn.functional as F

from fuzzy_relu import fuzzy_relu

ATTENTION_ITERATIONS = 3

class DagSage(nn.Module):
    """
    A graphsage inspired convolution on a directional acyclic graph
    """
    def __init__(self, input_dim, output_dim, representation_size, attention_iterations=ATTENTION_ITERATIONS):
        # input_dim: size of vector representation of incoming nodes
        # output_dim: size of node output dimension per node
        # representation_size: size of internal hidden layers
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_size = representation_size
        self.attention_iterations = attention_iterations
        self.node_self_rep = nn.Linear(input_dim, representation_size)
        self.src_representation = nn.Linear(output_dim, representation_size)
        self.src_keys = nn.Linear(output_dim, representation_size)
        self.forget_trans  = nn.Linear(representation_size*4, representation_size)
        self.hidden_trans = nn.Linear(representation_size*4, representation_size)
        self.query_gen = nn.Linear(representation_size*3, representation_size)
        self.out_trans = nn.Linear(representation_size*3, output_dim)

    def cuda(self):
        self.node_self_rep = self.node_self_rep.cuda()
        self.src_representation = self.src_representation.cuda()
        self.src_keys = self.src_keys.cuda()
        
        return self

    def forward(self, nodes_adj):
        # nodes_adj[0]: (batch, node, vector) - node representations
        # nodes_adj[1]: (batch, node^2)
        # nodes_adj[1] is a directional adjacency matrix

        batch = nodes_adj[0].shape[0]


        node_id_rep = self.node_self_rep(nodes_adj[0])

        num_nodes = nodes_adj[0].shape[1]
        out_nodes = torch.zeros((batch, num_nodes, self.output_dim))

        src_conn = nodes_adj[1]

        for i in range(num_nodes):
            hidden = torch.zeros((batch, self.representation_size))
            src_rep = self.src_representation(out_nodes)
            src_keys = self.src_keys(out_nodes)
            query = torch.zeros((batch, self.representation_size))
            # Minimal gated unit
            for j in range(self.attention_iterations):
                inp_att = torch.einsum('bk,bik->bi', query, src_keys)
                # Negate by max attention match to reduce chance of max clipping
                inp_att = (inp_att.t()-torch.max(inp_att, dim=1).values).t()
                inp_att = torch.exp(inp_att) # Softmax attention
                inp_att *= src_conn[:,:,i]
                inv_sum = 1/torch.sum(inp_att, dim=1)
                inp_att = torch.einsum('bi,b->bi', inp_att, inv_sum)
                query_res = torch.einsum('bi,biv->bv', inp_att, src_rep)
                forget = self.forget_trans(torch.cat((query_res, hidden, node_id_rep[:,i], query), dim=1))
                forget = torch.sigmoid(forget)
                hidden = forget*hidden + (1-forget)*torch.tanh(self.hidden_trans(torch.cat(query_res, forget*hidden, node_id_rep[:,i], query), dim=1))
                query = self.query_gen(torch.cat((hidden, node_id_rep[:,i], query), dim=1))
            out_nodes[:,i] = self.out_trans(torch.cat((hidden, node_id_rep[:,i], query), dim=1))
        return (out_nodes, nodes_adj[1])

class ReverseDagSage(nn.Module):
    def __init__(self, *args, **kwargs):
        self.dagsage = DagSage(*args, **kwargs)

    def cuda(self):
        self.dagsage = self.dagsage.cuda()
        return self

    def forward(self, node_adj):
        reversed_nodes = torch.flip(node_adj[0], (1,))
        reversed_adjs = torch.flip(node_adj[1], (0,1))
        res = self.dagsage((reversed_nodes, reversed_adjs))
        return (torch.flip(res[0], (1,)), node_adj[1])
