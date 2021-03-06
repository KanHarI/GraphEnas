
import math
import random

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F

import supergraph as sg
import graphsage as gs
from fuzzy_relu import FuzzyRelu


def conv3(channels):
    return nn.Conv2d(channels, channels, 3, padding=1)

def conv5(channels):
    return nn.Conv2d(channels, channels, 5, padding=2)

def dep_conv5(channels):
    return nn.Conv2d(channels, channels, 5, padding=2, groups=channels)

def dep_conv7(channels):
    return nn.Conv2d(channels, channels, 7, padding=3, groups=channels)

def diaconv3_2(channels):
    return nn.Conv2d(channels, channels, 3, padding=2, dilation=2)

def diaconv5_2(channels):
    return nn.Conv2d(channels, channels, 5, padding=4, dilation=2)

def maxpool3(*_):
    return nn.MaxPool2d(3, stride=1, padding=1)

def maxpool5(*_):
    return nn.MaxPool2d(5, stride=1, padding=2)

def maxpool7(*_):
    return nn.MaxPool2d(7, stride=1, padding=3)


ACTIVATIONS = [conv3, conv5, dep_conv5, dep_conv7, diaconv3_2, diaconv5_2, maxpool3, maxpool5, maxpool7]

SUPERGRAPH_MAX_SIZE = 64
MAX_HALVINGS = 8

GRAPHSAGE_LAYER_PER_DIM = 3
GRAPHSAGE_CHANNELS = 60
GRAPHSAGE_NUM_HALVINGS = math.ceil(math.log2(SUPERGRAPH_MAX_SIZE))

SUBMODEL_CHANNELS = 10
IMAGE_CHANNELS = 3

NODE_PREPROCESS_SIZE = 200

PAIR_SELECTOR_SIZE_0 = 200
PAIR_SELECTOR_SIZE_1 = 200
PAIR_SELECTOR_SIZE_2 = 100
PAIR_SELECTOR_SIZE_3 = 50
PAIR_SELECTOR_SIZE_4 = 10

NODE_PROCESSOR_SIZE_0 = 200
NODE_PROCESSOR_SIZE_1 = 50

CRITIC_SIZE_0 = 1000
CRITIC_SIZE_1 = 100
CRITIC_SIZE_2 = 10
CRITIC_OUTPUT_SIZE = 2


class Supermodel(nn.Module):
    def __init__(self, activations_list=ACTIVATIONS, max_size=SUPERGRAPH_MAX_SIZE, max_halvings=MAX_HALVINGS):
        super().__init__()
        self.max_size = max_size
        self.max_halvings = max_halvings
        self.log2_max_size = math.ceil(math.log2(max_size))
        self.activations_list = activations_list
        # +2 for input, output nodes, +2 for number of incoming and outgoing connections
        self.input_feature_sizes = 2 + len(activations_list) + self.log2_max_size*2 + max_halvings*2 +2

        self.graphsage_num_halvings = math.ceil(math.log2(max_size))

        self.actor_critic_graphsage = gs.BiPyramid(GRAPHSAGE_LAYER_PER_DIM, self.graphsage_num_halvings, GRAPHSAGE_CHANNELS)
        
        self.node_preprocessor = nn.Sequential(
            nn.Linear(self.input_feature_sizes, NODE_PREPROCESS_SIZE),
            FuzzyRelu(),
            nn.Linear(NODE_PREPROCESS_SIZE, GRAPHSAGE_CHANNELS))

        node_output_feature_sizes = len(activations_list)
        self.node_processor = nn.Sequential(
            nn.Linear(self.input_feature_sizes + GRAPHSAGE_CHANNELS - 3, NODE_PROCESSOR_SIZE_0),
            FuzzyRelu(),
            nn.Linear(NODE_PROCESSOR_SIZE_0, NODE_PROCESSOR_SIZE_1),
            FuzzyRelu(),
            nn.Linear(NODE_PROCESSOR_SIZE_1, node_output_feature_sizes))

        # inputs: +current distance, +1 for current connectedness
        # outputs: connectedeness [yes\no], priority
        self.pair_selector = nn.Sequential(
            nn.Linear(self.input_feature_sizes*2 + (GRAPHSAGE_CHANNELS-3)*2 + self.log2_max_size + 1, PAIR_SELECTOR_SIZE_0),
            FuzzyRelu(),
            nn.Linear(PAIR_SELECTOR_SIZE_0, PAIR_SELECTOR_SIZE_1),
            FuzzyRelu(),
            nn.Linear(PAIR_SELECTOR_SIZE_1, PAIR_SELECTOR_SIZE_2),
            FuzzyRelu(),
            nn.Linear(PAIR_SELECTOR_SIZE_2, PAIR_SELECTOR_SIZE_3),
            FuzzyRelu(),
            nn.Linear(PAIR_SELECTOR_SIZE_3, PAIR_SELECTOR_SIZE_4),
            FuzzyRelu(),
            nn.Linear(PAIR_SELECTOR_SIZE_4, 2 + 1)
            )

        self.critic = nn.Sequential(
            nn.Linear(GRAPHSAGE_CHANNELS*(1+self.graphsage_num_halvings), CRITIC_SIZE_0),
            FuzzyRelu(),
            nn.Linear(CRITIC_SIZE_0, CRITIC_SIZE_1),
            FuzzyRelu(),
            nn.Linear(CRITIC_SIZE_1, CRITIC_SIZE_2),
            FuzzyRelu(),
            nn.Linear(CRITIC_SIZE_2, CRITIC_OUTPUT_SIZE))

    def cuda(self):
        self.actor_critic_graphsage = self.actor_critic_graphsage.cuda()
        self.node_preprocessor = self.node_preprocessor.cuda()
        self.node_processor = self.node_processor.cuda()
        self.pair_selector = self.pair_selector.cuda()
        self.critic = self.critic.cuda()
        return self

    def create_submodel(self, submodel_size, layers_between_halvings, output_dim, channels=SUBMODEL_CHANNELS, inp_channels=IMAGE_CHANNELS):
        return Submodel(submodel_size, channels, self, layers_between_halvings, output_dim, inp_channels)


class ExplorationExploitationSoftmax(nn.Module):
    def __init__(self, init_expt):
        super().__init__()
        self.softmax = nn.Softmax(0)
        self.expt = torch.tensor(init_expt, dtype=torch.float)

    def cuda(self):
        self.softmax = self.softmax.cuda()
        self.expt = self.expt.cuda()
        return self

    def forward(self, inp):
        return self.softmax(self.expt * inp)


class Submodel(nn.Module):
    def __init__(self, size, channels, supermodel, layers_between_halvings, output_dim, inp_channels):
        super().__init__()
        self.size = size
        self.channels = channels
        self.supermodel = supermodel
        self.layers_between_halvings = layers_between_halvings
        self.supergraph = sg.Supergraph(size, channels, self.supermodel.activations_list, layers_between_halvings, inp_channels)
        self.adj_matrix = torch.zeros(size, size)
        self.softmax = ExplorationExploitationSoftmax(0)
        self.saved_pred = []

        # Build random initial connections
        for i in range(size-1):
            for j in range(i+1,size):
                self.adj_matrix[i,j] = random.randint(0,1)
            
        # All initialized to first possible activation function...
        self.chosen_activations = torch.zeros(size, dtype=torch.int)
        for i in range(1,size-1):
            self.chosen_activations[i] = random.randint(0,len(self.supermodel.activations_list)-1)

        self.subgraph = self.supergraph.create_subgraph(self.chosen_activations, self.adj_matrix)
        self.final_classifier = nn.Linear(channels*(2**((size-1)//layers_between_halvings)), output_dim)
        self.valuation = None

    def cuda(self):
        self.supergraph = self.supergraph.cuda()
        self.adj_matrix = self.adj_matrix.cuda()
        self.chosen_activations = self.chosen_activations.cuda()
        self.subgraph = self.subgraph.cuda()
        self.softmax = self.softmax.cuda()
        self.final_classifier = self.final_classifier.cuda()
        return self

    def refresh_subgraph(self):
        if self.valuation is None:
            pass

        nodes = torch.zeros(self.size, self.supermodel.input_feature_sizes)
        if torch.cuda.is_available():
            nodes = nodes.cuda()

        for i in range(self.size):
            # Locations
            if i == 0:
                nodes[i,0] = 1
            elif i == self.size-1:
                nodes[i,1] = 1
            else:
                nodes[i,2 + self.chosen_activations[i]] = 1

        # Add position pointers
        ptr_for = 2 + len(self.supermodel.activations_list)
        ptr_rev = ptr_for + self.supermodel.log2_max_size
        for i in range(self.size):
            for_rep = ('0'*self.supermodel.log2_max_size + bin(i)[2:])[-self.supermodel.log2_max_size:]
            rev_rep = ('0'*self.supermodel.log2_max_size + bin(self.size-1-i)[2:])[-self.supermodel.log2_max_size:]
            for j in range(self.supermodel.log2_max_size):
                if for_rep[j] == '1':
                    nodes[i, ptr_for + j] = 1
                if rev_rep[j] == '1':
                    nodes[i, ptr_rev + j] = 1

        # Add halving num
        ptr_for = ptr_rev + self.supermodel.log2_max_size
        ptr_rev = ptr_for + self.supermodel.max_halvings
        for i in range(self.size):
            nodes[i, ptr_for + i//self.layers_between_halvings] = 1
            nodes[i, ptr_rev + ((self.size-1) // self.layers_between_halvings) - i // self.layers_between_halvings] = 1

        for i in range(self.size-1):
            for j in range(i,self.size):
                if self.adj_matrix[i,j] == 1:
                    nodes[i,-2] += 1
                    nodes[j, -1] += 1


        # TODO: cound incoming and outgoing connections

        _adj_matrix = torch.stack([self.adj_matrix])
        _nodes = torch.stack([nodes])
        na1 = (_nodes, _adj_matrix)
        _nodes = self.supermodel.node_preprocessor(_nodes)

        graphsage_res = self.supermodel.actor_critic_graphsage.forwardA((_nodes, _adj_matrix))[0][0]
        priority = graphsage_res[:,-3:].t()
        graphsage_res = graphsage_res[:,:-3]

        # 3 options: select node, select edge by source, select edge by dest
        update_method = random.randint(0,2)
        action_log_prob = None
        if update_method == 0:
            cn_d = distributions.Categorical(self.softmax(priority[0,1:self.size-1]))
            cn = cn_d.sample()
            action_log_prob = cn_d.log_prob(cn)

            node_processor_inp = torch.cat([nodes[cn], graphsage_res[cn]], dim=-1)

            node_processor_out = self.softmax(self.supermodel.node_processor(node_processor_inp))
            node_processor_out = distributions.Categorical(node_processor_out)
            selected_activation = node_processor_out.sample()
            nodes[cn, 2 + self.chosen_activations[cn]] = 0
            self.chosen_activations[cn] = selected_activation
            nodes[cn, 2 + self.chosen_activations[cn]] = 1
            action_log_prob += node_processor_out.log_prob(selected_activation)

        else: # edge update
            if update_method == 1:
                src_d = distributions.Categorical(self.softmax(priority[1,:self.size-1]))
                src = src_d.sample()
                action_log_prob = src_d.log_prob(src)
                # TODO
            src = random.randint(0, self.size-2)
            dst = random.randint(src+1 ,self.size-1)
            nodes[src, -2] -= 1
            nodes[dst, -1] -= 1
            distance = torch.zeros(self.supermodel.log2_max_size)
            
            if torch.cuda.is_available():
                distance = distance.cuda()

            str_distance = ("0"*self.supermodel.log2_max_size + bin(dst-src)[2:])[-self.supermodel.log2_max_size:]
            for i in range(self.supermodel.log2_max_size):
                if str_distance[i] == '1':
                    distance[i] = 1
            c_conn = torch.stack((self.adj_matrix[src,dst],))
            edge_processor_inp = torch.cat([nodes[src], nodes[dst], graphsage_res[src], graphsage_res[dst], distance, c_conn])
            if torch.cuda.is_available():
                edge_processor_inp = edge_processor_inp.cuda()

            edge_processor_out = self.softmax(self.supermodel.pair_selector(edge_processor_inp))
            edge_processor_out = distributions.Categorical(edge_processor_out)
            selected_edge_conn = edge_processor_out.sample()
            if selected_edge_conn == 1:
                nodes[src, -2] += 1
                nodes[dst, -1] += 1
            self.adj_matrix[src, dst] = selected_edge_conn
            action_log_prob = edge_processor_out.log_prob(selected_edge_conn)

        adj_matrix = torch.stack([self.adj_matrix])
        _nodes = torch.stack([nodes])
        na2 = (_nodes, _adj_matrix)
        _nodes = self.supermodel.node_preprocessor(_nodes - na1[0])


        with torch.no_grad():
            critic_res = self.supermodel.actor_critic_graphsage.forwardB((_nodes, _adj_matrix))
            critic_res = self.supermodel.critic(critic_res)
            critic_mean = critic_res[0,0]

        actor_loss = action_log_prob * critic_mean

        self.subgraph = self.supergraph.create_subgraph(self.chosen_activations, self.adj_matrix)

        return actor_loss, na1, na2


    def forward(self, inp):
        # Average over last dimension
        inp = self.subgraph(inp).mean(-1).mean(-1)
        inp = self.final_classifier(inp)
        return inp
