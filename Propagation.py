import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

class Propagation(object):
    def __init__(self, G=None):
        if G is not None:
            self.graph = G
    
    def read_graphml(self, filename):
        self.graph = nx.read_graphml(filename)

    def weighted_model(self):
        inds = self.graph.in_degree(weight='data')

        probs = dict()
        for e in self.graph.edges(data=True):
            w = 1
            if len(e[2]):
                w = e[2]['data']
            probs[(e[0], e[1])] = w/inds[e[1]]

        return probs

    def multi_model(self, p_range = None):
        if p_range is None:
            p_range = [0.01, 0.02, 0.04, 0.08]
            p_range = [0.5]

        probs = dict()
        for e in self.graph.edges():
            probs[(e[0], e[1])] = random.choice(p_range)

        return probs

    def spread_IC(self, seed_set, MC, model = 'multi'):
        if model == 'weighted':
            probs = self.weighted_model()
        elif model == 'multi':
            probs = self.multi_model()

        sub = len(seed_set)
        spreads = []
        for _ in range(MC):
            activated = set(seed_set)
            for node in seed_set:
                es = self.graph.out_edges(node)
                for e in es:
                    if e[1] not in activated and random.random() < probs[e]:
                        activated.add(e[1])
            spreads.append(len(activated) - sub)
        return np.mean(spreads)

    def greedy(self, k, MC, model):
        seed_set = []
        for size in range(k):
            print('iteration', size)
            max_spread = -1
            for node in self.graph:
                if node not in seed_set:
                    temp_set = seed_set + [node]
                    node_spread = self.spread_IC(temp_set, MC, model)
                    if node_spread > max_spread:
                        max_spread = node_spread
                        max_node = node
            seed_set += [max_node]
        return seed_set




if __name__ == '__main__':
    random.seed(2018)


    G = nx.gn_graph(20)
    print(G.edges())
    ppg = Propagation()
    ppg.graph = G

    seed_set = ppg.greedy(5, 10, 'multi')

    console = []