import networkx as nx
import random
import numpy as np
# import matplotlib.pyplot as plt
import time
import sys

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

    def greedy(self, k, MC, probs, filename=None):
        seed_set = []
        meta = {}
        for size in range(k):
            start2it = time.time()
            print('iteration', size)
            print('Processed', end=' ')
            sys.stdout.flush()
            percent = len(self.graph)//10
            max_spread = -1
            for en, node in enumerate(self.graph):
                if en > 0 and not en % percent:
                    print("{}%".format(en/percent*10), end=' ')
                    sys.stdout.flush()
                if node not in seed_set:
                    temp_set = seed_set + [node]
                    node_spread = self.spread_IC(temp_set, MC, probs)
                    if node_spread > max_spread:
                        max_spread = node_spread
                        max_node = node
            print()
            seed_set += [max_node]
            finish2it = time.time()
            meta[size] = [max_spread, finish2it - start2it]
            if filename is not None:
                with open(filename, 'a+') as f:
                    f.write(f"{max_node} {max_spread} {finish2it - start2it}\n")
        return seed_set, meta

    def greedy_emb(self, k, MC, probs, candidates, seed_set, filename = None):
        meta = {}
        for size in range(k):
            start2it = time.time()
            print('iteration', size)
            print('Processed', end=' ')
            sys.stdout.flush()
            percent = len(self.graph)//10
            max_spread = -1
            for en, node in enumerate(candidates):
                if en > 0 and not en % percent:
                    print("{}%".format(en/percent*10), end=' ')
                    sys.stdout.flush()
                if node not in seed_set:
                    temp_set = seed_set + [node]
                    node_spread = self.spread_IC(temp_set, MC, probs)
                    if node_spread > max_spread:
                        max_spread = node_spread
                        max_node = node
            print()
            seed_set += [max_node]
            finish2it = time.time()
            meta[size] = [max_spread, finish2it - start2it]
            if filename is not None:
                with open(filename, 'a+') as f:
                    f.write(f"{max_node} {max_spread} {finish2it - start2it}\n")
        return seed_set, meta




if __name__ == '__main__':
    random.seed(2018)

    G = nx.read_edgelist('../Data/grqc_cpp.txt', create_using=nx.DiGraph())
    print(type(G))
    print(len(G), len(G.edges()))

    def write_meta(filename, meta):
        with open(filename, 'w') as f:
            for line in sorted(meta.items()):
                f.write(f"{line[0]}, {line[1][0]}, {line[1][1]}\n")


    ppg = Propagation(G=G)
    probs = ppg.weighted_model()

    # Experiment 1: standard greedy





    seed, meta = ppg.greedy(k=20, MC=100, probs=probs, filename='greedy_full.txt')
    print('Seed set:', seed)
    print('Meta:', meta)
    with open('grqc_greedy_seed.txt', 'w+') as f:
        for node in seed:
            f.write(f"{node}\n")
    write_meta("grqc_greedy_results.csv", meta)

    # Experiment 2: greedy with candidates embeddings
    # top = []
    # with open('../Data/classifier_ranging.txt') as f:
    #     for line in f:
    #         top.append(str(int(line)))
    #         if len(top) == 200:
    #             break
    # S = []
    with open('../Data/Wiki_final_ranging.txt') as f:
        for line in f:
            S.append(str(int(line)))
            if len(S) == 10:
                break
    seed, meta = ppg.greedy_emb(k=10, MC=100, probs=probs, candidates=top, seed_set=S, filename="greedy_candidates_full.txt")
    print('Seed set:', seed)
    print('Meta:', meta)
    with open('candidates_seed.txt', 'w+') as f:
        for node in seed:
            f.write(f"{node}\n")
    write_meta("candidates_results.csv", meta)

    console = []