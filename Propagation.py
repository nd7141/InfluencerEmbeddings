import networkx as nx
import random
import matplotlib.pyplot as plt

class Propagation(object):
    def __init__(self):
        pass
    
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

        probs = dict()
        for e in self.graph.edges():
            probs[(e[0], e[1])] = random.choice(p_range)

        return probs

    def runIC(self, seed_set):
        pass



if __name__ == '__main__':
    G = nx.gn_graph(10)

    ppg = Propagation()
    ppg.graph = G

    console = []