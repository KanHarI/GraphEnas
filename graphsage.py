
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import copy


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
        # graph_nodes_batch: (batch, node, vector)
        # graph_adj_batch: (batch, node^2)
        # graph_adj_batch is a directional adjacency matrix, can accept non-binary inputs

        # Aggregation may replaced by smarter aggregation in the future.
        # For now it is sum for simplicity and efficiency.
        
        src_representation = self.src_representation(nodes_adj[0])
        conn = F.normalize(nodes_adj[1], dim=2)
        src_representation = torch.einsum('bjv,bij->biv', (src_representation, conn))


        dst_representation = self.dst_representation(nodes_adj[0])
        conn = F.normalize(nodes_adj[1], dim=1)
        dst_representation = torch.einsum('bjv,bji->biv', (dst_representation, conn))

        
        # src_representation = F.relu(src_representation)
        # node_id_rep = F.relu(self.node_self_rep(nodes_adj[0]))
        node_id_rep = self.node_self_rep(nodes_adj[0])
        # dst_representation = F.relu(dst_representation)

        update_src = torch.cat((src_representation, node_id_rep, dst_representation), dim=2)
        res = F.relu(self.node_update(update_src))
        return res


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
            # if batchnorm:
            #     self.norm_layers.append(nn.BatchNorm2d(1, momentum=0.01))
                

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
            fpass_graph = self.layers[i]((fpass_graph, adj))
        return fpass_graph

