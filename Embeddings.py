from __future__ import division

import networkx as nx
import random, time, math, os, sys
import numpy as np
from collections import Counter

class AnonymousWalks(object):
    '''
    Computes Anonymous Walks of a Graph.
    Class has a method to embed a graph into a vector space using anonymous walk distribution.
    Additionally, it has methods to do a sampling of anonymous walks, calculate possible
    anonymous walks of length l, generate a random batch of anonymous walks for AWE distributed model,
    and other utilities.
    '''
    def __init__(self, G = None):
        self.graph = G
        # paths are dictionary between step and all-paths
        self.paths = dict()
        self.__methods = ['sampling', 'exact']

    def read_graph_from_text(self, filename, header = True, weights = True, sep = ',', directed = False):
        '''Read from Text Files.'''
        G = nx.Graph()
        if directed:
            G = nx.DiGraph()
        with open(filename) as f:
            if header:
                next(f)
            for line in f:
                splitted = line.strip().split(sep)
                u = splitted[0]
                v = splitted[1]
                G.add_edge(u, v)
                if weights:
                    w = float(splitted[2])
                    G[u][v]['weight'] = w
        self.graph = G
        return self.graph

    def read_graphml(self, filename):
        '''Read graph from graphml format.'''
        self.graph = nx.read_graphml(filename)
        return self.graph

    def create_random_walk_graph(self):
        '''Creates a probabilistic graph from graph.
        If edges have parameter "weight" then it will use the weights in computing probabilities.'''
        if self.graph is None:
            raise ValueError("You should first create a weighted graph.")

        # get name of the label on graph edges (assume all label names are the same)
        label_name = 'weight'
        # for e in self.graph.edges_iter(data=True):
        #     label_name = e[2].keys()[0]
        #     break

        RW = nx.DiGraph()
        for node in self.graph:
            edges = self.graph[node]
            total = float(sum([edges[v].get(label_name, 1) for v in edges if v != node]))
            for v in edges:
                if v != node:
                    RW.add_edge(node, v, weight = edges[v].get(label_name,1) / total)
        self.rw_graph = RW

    def _all_paths(self, steps, keep_last = True):
        '''Get all possible anonymous walks of length up to steps.'''
        paths = []
        last_step_paths = [[0, 1]]
        for i in range(2, steps+1):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if walks[-1] != j and j <= max(walks) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        # filter only on n-steps walks
        if keep_last:
            paths = list(filter(lambda path: len(path) ==  steps + 1, paths))
        self.paths[steps] = paths

    def walk2pattern(self, walk):
        '''Converts a walk with arbitrary nodes to anonymous walks, without considering labels.'''
        idx = 0
        pattern = []
        d = dict()
        for node in walk:
            if node not in d:
                d[node] = idx
                idx += 1
            pattern.append(d[node])
        return tuple(pattern)

    def n_samples(self, steps, delta, eps):
        '''Number of samples with eps and delta concetration inequality.'''
        a = len(list(self.paths[steps]))
        estimation = 2*(math.log(2)*a + math.log(1./delta))/eps**2
        return int(estimation) + 1

    def _random_step_node(self, node):
        '''Moves one step from the current according to probabilities of outgoing edges.
        Return next node.'''
        if self.rw_graph is None:
            raise ValueError("Create a Random Walk graph first with {}".format(self.create_random_walk_graph.__name__))
        r = random.uniform(0, 1)
        low = 0
        if len(self.rw_graph[node]):
            for v in self.rw_graph[node]:
                p = self.rw_graph[node][v]['weight']
                if r <= low + p:
                    return v
                low += p
        else:
            return node # return same node if there are no outgoing edges


    def anonymous_walk(self, node, steps):
        '''Creates anonymous walk from a node for arbitrary steps.
        Returns a tuple with consequent nodes.'''
        d = dict()
        d[node] = 0
        count = 1
        walk = [d[node]]
        for i in range(steps):
            v = self._random_step_node(node)
            if v not in d:
                d[v] = count
                count += 1
            walk.append(d[v])
            node = v
        return tuple(walk)

    def _sampling(self, steps, MC):
        '''Find anonymous walk distribution using sampling approach for a node.
        Run MC random walks for random nodes in the graph.
        steps is the number of steps.
        MC is the number of iterations.
        Returns dictionary pattern to probability.
        '''
        node_aw = dict()
        for node in self.rw_graph:
            walks = Counter()
            for it in range(MC):
                w = self.anonymous_walk(node, steps)
                walks[w] += 1./MC
            node_aw[node] = walks
        return node_aw

    def _exact(self, steps, labels = None, verbose = True):
        '''Find anonymous walk distribution exactly for a node.
        Calculates probabilities from each node to all other nodes within n steps.
        Running time is the O(# number of random walks) <= O(n*d_max^steps).
        labels, possible values None (no labels), 'edges', 'nodes', 'edges_nodes'.
        steps is the number of steps.
        Returns dictionary pattern to probability.
        '''
        def patterns(RW, node, steps, walks, current_walk=None, current_dist=1.):
            if current_walk is None:
                current_walk = [node]
            if len(current_walk) > 1:  # walks with more than 1 edge
                all_walks.append(current_walk)
                if labels is None:
                    w2p = self.walk2pattern(current_walk)
                else:
                    raise ValueError('labels argument should be one of the following: edges, nodes, edges_nodes, None.')
                amount = current_dist
                walks[w2p] = walks.get(w2p, 0) + amount # / len(RW) test: not normalizing
            if steps > 0:
                for v in RW[node]:
                    patterns(RW, v, steps - 1, walks, current_walk + [v], current_dist * RW[node][v]['weight'])


        node_aw = dict()
        for node in self.rw_graph:
            walks = dict()
            all_walks = []
            patterns(self.rw_graph, node, steps, walks)
            node_aw[node] = walks
        if verbose:
            print('Total walks of size {} in a graph:'.format(steps), len(all_walks))
        return node_aw

    def embed(self, steps, method = 'exact', MC = None, delta = 0.1, eps = 0.1,
              labels = None, verbose = True):
        '''Get embeddings of a graph using anonymous walk distribution.
        method can be sampling, exact
        steps is the number of steps.
        MC is the number of iterations.
        labels, possible values None (no labels), 'edges', 'nodes', 'edges_nodes'.
        delta is probability devitation from the true distribution of anonymous walks
        eps is absolute value for deviation of first norm
        Return vector and meta information as dictionary.'''

        # Create a random walk instance of the graph first
        self.create_random_walk_graph()

        if labels is None:
            self._all_paths(steps)
        else:
            raise ValueError('labels argument should be one of the following: edges, nodes, edges_nodes, None.')

        if method == 'sampling':
            if verbose:
                print("Use sampling method to get vector representation.")
            if MC is None:
                MC = self.n_samples(steps, delta, eps)
                if verbose:
                    print("Using number of iterations = {} for delta = {} and eps = {}".format(MC, delta, eps))
            start = time.time()
            node_aw = self._sampling(steps, MC)
            finish = time.time()
            if verbose:
                print('Spent {} sec to get vector representation via sampling method.'.format(round(finish - start, 2)))
        elif method == 'exact':
            if verbose:
                print("Use exact method to get vector representation.")
            start = time.time()
            node_aw = self._exact(steps, labels = labels, verbose=verbose)
            finish = time.time()
            if verbose:
                print('Spent {} sec to get vector representation via exact method.'.format(round(finish - start, 2)))
        else:
            raise ValueError("Wrong method for AnonymousWalks.\n You should choose between {} methods".format(', '.join(self.__methods)))

        E = np.zeros((len(self.rw_graph), len(self.paths[steps])))
        nodes_for_emb_matrix = []
        for i, node in enumerate(self.rw_graph):
            nodes_for_emb_matrix.append(node)
            vector = []
            for path in self.paths[steps]:
                vector.append(node_aw[node].get(tuple(path), 0))
            E[i, :] = vector
        return E, nodes_for_emb_matrix, {'meta-paths': self.paths[steps]}


if __name__ == '__main__':
    random.seed(2018)

    G = nx.gn_graph(20)
    G = nx.erdos_renyi_graph(20, 0.5)
    print(G.edges())

    aw = AnonymousWalks(G=G)
    E, nodes, meta = aw.embed(3, 'exact', 10)
    print(E)
    print(len(nodes))
    print(meta)

    console = []
