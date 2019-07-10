# GraphEnas
Graph convolution based neural architecture search with weight sharing.
This work was inspired by the Efficient Neural Architecture Search via Weight Sharing paper,
and draws from the field of graph convlutions as a way to calculate graph modifying actions.

Unfortunately, the training phase seems to be unreasonably slow and unstable - probably the graph
convolution having a single data point for every training batch of the network is part of the blame.

#Theory
This algorithm uses a "Supergraph" DAG in which every node is a layer of the neural network, the operation in whom is encoded as a vector of possible layers (3x3 & 5x5 convolutions, depthwise 5x5 & 7x7 convolutions, dialated 3x3 & 5x5 convolutions, 3x3 & 5x5 & 7x7 max pools) with "1" in the current active layer and "0" in the other layers. The connections between the various layers of the graph is implemented by a diagonal matrix containing "1"s at layer connections.

At every stage, an actor chooses a graph modifying action, and a critic tries to guess the performence of the network on a part of the test set after the given change. The neural net is then trained for a fixed amount of time (time is chosen as otherwise the NN can degrade into allowing all possible skip connecitons, as this is a stronger yet much too slow connection). The critic loss is the delta between the actual network performence and the predicted network performance, and the actor loss is based upon it's performence as predicted by the critic.

At every training batch, the shared wights are updated as in the original "Efficient Neural Architecture Search via Weight Sharing" paper.
