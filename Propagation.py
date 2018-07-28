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
        inds = self.graph.in_degree()

        probs = dict()
        for e in self.graph.edges(data=True):
            probs[(e[0], e[1])] = 1/inds[e[1]]

        return probs

    def multi_model(self, p_range = None):
        if p_range is None:
            p_range = [0.01, 0.02, 0.04, 0.08]

        probs = dict()
        for e in self.graph.edges():
            probs[(e[0], e[1])] = random.choice(p_range)

        return probs

    def spread_IC(self, seed_set, MC, probs):
        spreads = []
        for _ in range(MC):
            activated = seed_set.copy()
            for node in activated:
                es = self.graph.out_edges(node)
                for e in es:
                    if e[1] not in activated and random.random() < probs[e]:
                        activated.append(e[1])
            spreads.append(len(activated))
        return np.mean(spreads)

    def greedy(self, k, MC, probs):
        seed_set = []
        for size in range(k):
            print('iteration', size)
            print('Processed', end=' ')
            percent = len(self.graph)//10
            max_spread = -1
            for en, node in enumerate(self.graph):
                if en > 0 and not en % percent:
                    print("{}%".format(en/percent*10), end=' ')
                if node not in seed_set:
                    temp_set = seed_set + [node]
                    node_spread = self.spread_IC(temp_set, MC, probs)
                    if node_spread > max_spread:
                        max_spread = node_spread
                        max_node = node
            print()
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