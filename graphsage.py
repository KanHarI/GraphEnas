
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import copy
import math

import biqueue

class GraphSageLayer(nn.Module):
    def __init__(self, input_dim, output_dim, representation_size):
        # input_dim: size of vector representation of incoming nodes
        # output_dim: size of node output dimension per node
        # representation_size: size of internal hidden layers
        #
        #
        #               --find all incoming edges -> in_to_representation of source nodes--
        #              /                                                                   \
        # input_nodes -----node_to_rep-----------------------------------------------------CONCAT---node_update
        #              \                                                                   /
        #               --find all outgoing edges -> out_to_representation of source nodes-
        #
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_size = representation_size
        self.src_representation = nn.Linear(input_dim, representation_size)
        self.dst_representation = nn.Linear(input_dim, representation_size)
        self.node_self_rep = nn.Linear(input_dim, representation_size)
        self.node_update = nn.Linear(3*representation_size, output_dim)

    def cuda(self):
        self.src_representation = self.src_representation.cuda()
        self.dst_representation = self.dst_representation.cuda()
        self.node_self_rep = self.node_self_rep.cuda()
        self.node_update = self.node_update.cuda()
        return self

    def forward(self, nodes_adj):
        # nodes_adj[0]: (batch, node, vector)
        # nodes_adj[1]: (batch, node^2)
        # nodes_adj[1] is a directional adjacency matrix, can accept non-binary inputs

        src_representation = self.src_representation(nodes_adj[0])
        conn = F.normalize(nodes_adj[1], dim=2)
        src_representation = torch.einsum('bjv,bij->biv', (src_representation, conn))


        dst_representation = self.dst_representation(nodes_adj[0])
        conn = F.normalize(nodes_adj[1], dim=1)
        dst_representation = torch.einsum('bjv,bji->biv', (dst_representation, conn))

        node_id_rep = self.node_self_rep(nodes_adj[0])

        update_src = torch.cat((src_representation, node_id_rep, dst_representation), dim=2)
        res = torch.tanh(self.node_update(update_src))
        return (res, nodes_adj[1])


class GraphPoolLayer(nn.Module):
    # A max pool layer of a graph
    # Clusters every pair of succeeding nodes in the representation togather
    # ASSUMES MEANINGFULL NODE ORDER, AND DAG.
    def __init__(self, size_factor, input_dim, output_dim):
        super().__init__()
        self._1x1conv = nn.Linear(input_dim, output_dim)
        self.nodes_pool = nn.MaxPool1d(size_factor, ceil_mode=True)
        self.adj_pool = nn.MaxPool2d(size_factor, ceil_mode=True)

    def cuda(self):
        self._1x1conv = self._1x1conv.cuda()
        self.nodes_pool = self.nodes_pool.cuda()
        self.adj_pool = self.adj_pool.cuda()
        return self

    def forward(self, nodes_adj):
        nodes = self._1x1conv(nodes_adj[0])
        nodes = self.nodes_pool(nodes.transpose(1,2)).transpose(1,2)
        adj = torch.stack((nodes_adj[1],), 1)
        adj = self.adj_pool(adj)[:,0,:,:]
        return (nodes, adj)


class GraphUnpoolLayer(nn.Module):
    def __init__(self, size_factor, input_dim, output_dim):
        super().__init__()
        self.nodes_unpool = nn.ConvTranspose1d(input_dim, output_dim, size_factor, size_factor)

    def cuda(self):
        self.nodes_unpool = self.nodes_unpool.cuda()
        return self

    def forward(self, nodes_adj):
        nodes = nodes_adj[0]
        nodes = self.nodes_unpool(nodes.transpose(1,2)).transpose(1,2)
        return (nodes, None) # Cannot reconstruct adj. matrix


class BiPyramid(nn.Module):
    # This is a graph network with skip connections and 2 outputs:
    # One vector and one per graph node
    # (This assumes there are enough layers of pooling to reach a single vector)
    #
    # Input
    # |
    # L0
    # | \
    # |  L1
    # |  | \
    # |  |  L2
    # |  |  | \
    # |  |  | Maxpool
    # |  |  |  | \
    # |  |  |  |  L3
    # |  |  |  |  | \
    # |  |  |  |  |  L4
    # |  |  |  |  | /|
    # |  |  |  |  L5 \
    # |  |  |  | /  \ \
    # |  |  | Unpool| |
    # |  |  | /|    / /
    # |  |  L6 |   / /
    # |  | /|  |  /  |
    # |  L7 |  |  |  |
    # | /|  |  |  |  |
    # L8 |  |  |  |  |
    # | \|  |  |  |  |
    # | *L9 |  |  |  | <- The graph is modified at the input here
    # |    \|  |  |  |
    # |     L10|  |  |
    # |       \|  |  |
    # |    Maxpool|  |
    # |          \|  |
    # |           L12|
    # |             \|
    # |              L13
    # |               |
    # O1              O2
    #
    def __init__(self, layers_per_dim, num_halvings, channels):
        super().__init__()
        self.layers_1 = []
        self.layers_2 = []
        self.layers_3 = []
        self.links_12 = []
        self.links_13 = []
        self.links_23 = []
        # Build downstream ladder
        for i in range(num_halvings+1):
            _channels = channels*(2**i)
            for j in range(layers_per_dim):
                self.layers_1.append(GraphSageLayer(_channels, _channels, _channels))
            if i < num_halvings:
                self.layers_1.append(GraphPoolLayer(2, _channels, _channels*2))
        # Build upstream ladder
        for i in range(num_halvings+1):
            _channels = channels*(2**(num_halvings-i))
            if i > 0:
                self.links_12.append(nn.Linear(2*_channels, 2*_channels))
                self.layers_2.append(GraphUnpoolLayer(2, 2*_channels, _channels))
            for j in range(layers_per_dim):
                self.links_12.append(nn.Linear(_channels, _channels))
                self.layers_2.append(GraphSageLayer(_channels, _channels, _channels))
        # Build 2nd downstream ladder
        for i in range(num_halvings+1):
            _channels = channels*(2**i)
            for j in range(layers_per_dim):
                self.links_13.append(nn.Linear(_channels, _channels))
                self.links_23.append(nn.Linear(_channels, _channels))
                self.layers_3.append(GraphSageLayer(_channels, _channels, _channels))
            if i < num_halvings:
                self.links_13.append(nn.Linear(_channels, _channels))
                self.links_23.append(nn.Linear(_channels, _channels))
                self.layers_3.append(GraphPoolLayer(2, _channels, _channels*2))
        
        self.stash = None

    def cuda(self):
        self.layers_1 = list(map(lambda x: x.cuda(), self.layers_1))
        self.layers_2 = list(map(lambda x: x.cuda(), self.layers_2))
        self.layers_3 = list(map(lambda x: x.cuda(), self.layers_3))
        self.links_12 = list(map(lambda x: x.cuda(), self.links_12))
        self.links_13 = list(map(lambda x: x.cuda(), self.links_13))
        self.links_23 = list(map(lambda x: x.cuda(), self.links_23))
        return self


    def forward(self, nodes_adj):
        self.stash = biqueue.Biqueue()
        # Downstream
        for l in self.layers_1:
            nodes_adj = l(nodes_adj)
            self.stash.push_back(nodes_adj)

        # Upstream
        for i,l in enumerate(self.layers_2):
            adj = self.stash.get(-1-i)[1]
            nodes = nodes_adj[0] + self.links_12[i](self.stash.get(-1-i)[0])
            nodes = nodes[:,:adj.shape[1],:]
            nodes_adj = (nodes, adj)
            nodes_adj = l(nodes_adj)
            self.stash.push_front(nodes_adj)

        return nodes_adj

    # This function is seperated from "forward" to allow modifying the computational graph
    def f2(self, nodes_adj):
        # Downstream
        for i,l in enumerate(self.layers_2):
            adj = nodes_adj[1]
            nodes = nodes_adj[0] + self.links_13[i](self.stash.get(-1-i)) + self.links_23[i](self.stash.get(i))
            nodes_adj = (nodes, adj)
            nodes_adj = l(nodes_adj)

        return nodes_adj[0].mean(1)


class PyramidGraphSage(nn.Module):
    # This architecture allows for skip connections:
    # (Example of layout with 8 layers)
    # Input
    # | \
    # |  \
    # |   L0
    # |  /| \
    # | / |  \
    # | | |   L1
    # | | |  /| \
    # | | | / |  \
    # | | | | |   L2
    # | | | | |  /| \
    # | | | | | / |  \
    # | | | | | | |   L3
    # | | | | | | |  /
    # | | | | | | | /
    # | | | | | | L4
    # | | | | | | |
    # | | | | \ |/
    # | | | |  L5
    # | | | | /
    # | | \ |/
    # | |  L6
    # | | /
    # \ |/
    #  L7
    #  |
    # Output
    #
    # This allows efficient training with "Lazy layer training":
    # I->L7,
    # I->L0->L7,
    # I->L0->L6->L7,
    # I->L0->L1->L6->L7...
    # Effectively "training one layer at a time" continously

    def __init__(self, num_layers, feature_sizes, representation_sizes=None):
        assert num_layers%2 == 0
        assert num_layers == len(feature_sizes)-1
        super().__init__()
        self.feature_sizes = feature_sizes
        self.num_layers = num_layers
        if representation_sizes is None:
            representation_sizes = feature_sizes[:-1]
        self.layers = []
        self.norm_layers = []
        for i in range(self.num_layers):
            if i < self.num_layers//2:
                self.layers.append(GraphSageLayer(
                    feature_sizes[i],
                    feature_sizes[i+1],
                    representation_sizes[i]))
            elif i == self.num_layers//2:
                self.layers.append(GraphSageLayer(
                    feature_sizes[i]+feature_sizes[self.num_layers-i-1],
                    feature_sizes[i+1],
                    representation_sizes[i]))
            else:
                self.layers.append(GraphSageLayer(
                    feature_sizes[i]+feature_sizes[self.num_layers-i]+feature_sizes[self.num_layers-i-1],
                    feature_sizes[i+1],
                    representation_sizes[i]))
                

    def cuda(self):
        self.layers = list(map(lambda x: x.cuda(), self.layers))
        self.norm_layers = list(map(lambda x: x.cuda(), self.norm_layers))
        return self

    def forward(self, nodes_adj):
        fpass_graph = nodes_adj[0]
        adj = nodes_adj[1]
        stashed_results = []
        for i in range(self.num_layers):
            if i < self.num_layers//2:
                stashed_results.append(fpass_graph)
            elif i == self.num_layers//2:
                # Concatenate skip connection inputs for pyramid "downward slope"
                fpass_graph = torch.cat((fpass_graph, stashed_results[self.num_layers-i-1]), dim=2)
            else:
                # Concatenate skip connection inputs for pyramid "downward slope"
                fpass_graph = torch.cat((fpass_graph, stashed_results[self.num_layers-i], stashed_results[self.num_layers-i-1]), dim=2)
            fpass_graph = self.layers[i]((fpass_graph, adj))[0]
        return fpass_graph


