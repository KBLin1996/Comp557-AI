from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import numpy as np

# create the nodes
a = BbnNode(Variable(0, 'cloudy', ['on', 'off']), [0.5, 0.5])
b = BbnNode(Variable(1, 'sprinkler', ['on', 'off']), [0.1, 0.9, 0.5, 0.5])
c = BbnNode(Variable(2, 'rain', ['on', 'off']), [0.8, 0.2, 0.2, 0.8])
d = BbnNode(Variable(3, 'wetgrass', ['on', 'off']), [0.99, 0.01, 0.9, 0.1, 0.9, 0.1, 0.00, 1])

# create the network structure
bbn = Bbn() \
    .add_node(a) \
    .add_node(b) \
    .add_node(c) \
    .add_node(d) \
    .add_edge(Edge(a, b, EdgeType.DIRECTED)) \
    .add_edge(Edge(a, c, EdgeType.DIRECTED)) \
    .add_edge(Edge(b, d, EdgeType.DIRECTED)) \
    .add_edge(Edge(c, d, EdgeType.DIRECTED))

# convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# insert an observation evidence
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('cloudy')) \
    .with_evidence('on', 1.0) \
    .build()
join_tree.set_observation(ev)
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('wetgrass')) \
    .with_evidence('on', 1.0) \
    .build()
join_tree.set_observation(ev)

# print the marginal probabilities
# for node in join_tree.get_bbn_nodes():
#     potential = join_tree.get_bbn_potential(node)
#     print(node)
#     print(potential)

Q = np.array([[0,       0.4455, 0.5545, 0],
              [0.0545,  0.9455, 0,      0],
              [0.4074,  0,      0.5926, 0],
              [0,       0.5,    0.5,    0]])

A = Q
# Stationary distribution of transition kernel
init = np.array([1, 0, 1, 1])
for i in range(1000):
    A = np.dot(A, Q)
print(A[0])

for i in range(5000):
    A = np.dot(A, Q)
print(A[0])

for i in range(10000):
    A = np.dot(A, Q)
print(np.dot(init, A[0]))