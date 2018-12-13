
import supergraph as sg
import graphsage as gs

import torch
import torch.nn as nn
import math
import random


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


GRAPHSAGE_LAYERS = 20
GRAPHSAGE_REPRESENTATION_SIZE = 20

SUBMODEL_CHANNELS = 10

class Supermodel(nn.Module):
	def __init__(self, graphsage_conv_layers=GRAPHSAGE_LAYERS, activations_list=ACTIVATIONS, max_size=256, max_halvings=8):
		super().__init__()
		self.activations_list = activations_list
		input_feature_sizes = len(activations_list) + math.ceil(math.log2(max_size)) + max_halvings
		self.graphsage = gs.PyramidGraphSage(
			graphsage_conv_layers,
			[input_feature_sizes] + [GRAPHSAGE_REPRESENTATION_SIZE] * graphsage_conv_layers,
			[GRAPHSAGE_REPRESENTATION_SIZE] * graphsage_conv_layers)
		
		# +1 for priority, + max_halvings for amount of dimensional halvings
		node_output_feature_sizes = len(activations_list) + 1 + max_halvings
		self.node_processor = nn.Linear(input_feature_sizes + GRAPHSAGE_REPRESENTATION_SIZE, node_output_feature_sizes)

		# inputs: +1 for current connectedness, 
		# outputs: priority, connectedeness [yes\no]
		self.pair_selector = nn.Linear(GRAPHSAGE_REPRESENTATION_SIZE*2 + math.ceil(math.log2(max_size)) + 1, 1 + 2)

	def cuda(self):
		self.graphsage = self.graphsage.cuda()
		self.node_processor = self.node_processor.cuda()
		self.pair_selector = self.pair_selector.cuda()
		return self

	def create_submodel(self, submodel_size, layers_between_halvings, output_dim, channels=SUBMODEL_CHANNELS, inp_channels=IMAGE_CHANNELS):
		return Submodel(submodel_size, channels, self.graphsage, self.node_processor, self.pair_selector, self.activations_list, layers_between_halvings, output_dim, inp_channels)


IMAGE_CHANNELS = 3

class Submodel(nn.Module):
	def __init__(self, size, channels, graphsage, node_processor, pair_selector, activations_list, layers_between_halvings, output_dim, inp_channels):
		super().__init__()
		self.size = size
		self.channels = channels
		self.graphsage = graphsage
		self.node_processor = node_processor
		self.pair_selector = pair_selector
		self.activations_list = activations_list
		self.layers_between_halvings = layers_between_halvings
		self.supergraph = sg.Supergraph(size, channels, activations_list, layers_between_halvings, inp_channels)
		self.adj_matrix = torch.zeros(size, size)

		# Build skip connections automatically
		for i in range(0,size-1):
			self.adj_matrix[i,i+1] = 1
			if i < size-i-1:
				self.adj_matrix[i,size-i-1] = 1

		# All initialized to first possible activation function...
		self.nodes = torch.zeros(size, dtype=torch.int)
		for i in range(size):
			self.nodes[i] = random.randint(0,len(activations_list)-1)

		self.subgraph = self.supergraph.create_subgraph(self.nodes, self.adj_matrix)
		self.final_classifier = nn.Linear(channels*(2**((size-1)//layers_between_halvings)), output_dim)

	def cuda(self):
		self.supergraph = self.supergraph.cuda()
		self.adj_matrix = self.adj_matrix.cuda()
		self.nodes = self.nodes.cuda()
		self.subgraph = self.subgraph.cuda()
		self.final_classifier = self.final_classifier.cuda()
		return self

	def refresh_subgraph(self):
		raise NotImplementedError()

	def forward(self, inp):
		inp = self.subgraph(inp).mean(-1).mean(-1)
		inp = self.final_classifier(inp)
		return inp
