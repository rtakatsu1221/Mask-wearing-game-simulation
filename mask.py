#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:24:55 2021

@author: rikuyatakatsu
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
  
G = nx.watts_strogatz_graph(n = 1000, k = 4, p = 0.5)
G_sf = nx.to_undirected(nx.scale_free_graph(n = 1000))

plt.figure(figsize = (12, 12))
nx.draw(G_sf)

def initGame(graph, thres):
    for node in graph.nodes():
        if (np.random.uniform(0, 1) < thres):
            strategy = 1 # mask
        else:
            strategy = 2 # no mask
        graph.nodes[node]['strategy'] = strategy
        graph.nodes[node]['payoff'] = 0

def getPayoff(graph, node, T, S):
    for v in list(graph.adj[node]):
        if (graph.nodes[node]['strategy'] == 1):
            if (graph.nodes[v]['strategy'] == 1):
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1
            else:
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + S
        else:
            if (graph.nodes[v]['strategy'] == 1):
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + T
    
def simulate(niter, T, S, graph, thres):
    initGame(graph, thres) 
    mask = list()
    
    for i in range(niter):
        count = 0
        for node in graph.nodes():
            if (graph.nodes[node]['strategy'] == 1):
                count = count + 1
        mask.append(count/len(graph.nodes()))
        u = np.random.randint(len(graph.nodes()))
        getPayoff(graph, u, T, S)
        v = np.random.randint(len(list(graph.adj[u])))
        getPayoff(graph, list(graph.adj[u])[v], T, S)
        if (graph.nodes[v]['payoff'] > graph.nodes[u]['payoff']):
            if (np.random.uniform(0, 1) < (graph.nodes[v]['payoff'] - graph.nodes[u]['payoff'])/(4*T)):
                graph.nodes[u]['strategy'] = graph.nodes[v]['strategy']
    return mask   
        
maskSW = simulate(50000, 1.9, 0.7, G, 0.25)
maskSW2 = simulate(50000, 1.9, 0.7, G, 0.5)
maskSW4 = simulate(50000, 1.9, 0.7, G, 0.75)
maskSW5 = simulate(50000, 1.9, 0.7, G, 0.85)
maskSW3 = simulate(50000, 1.9, 0.7, G, 0.9)
plt.figure(figsize = (8, 8))
plt.plot(range(50000), maskSW) 
plt.plot(range(50000), maskSW2) 
plt.plot(range(50000), maskSW4) 
plt.plot(range(50000), maskSW5)
plt.plot(range(50000), maskSW3) 

maskSF = simulate(200000, 1.9, 0.7, G_sf, 0.25)
maskSF2 = simulate(200000, 1.9, 0.7, G_sf, 0.5)
maskSF4 = simulate(200000, 1.9, 0.7, G_sf, 0.75)
maskSF5 = simulate(200000, 1.9, 0.7, G_sf, 0.85)
maskSF3 = simulate(200000, 1.9, 0.7, G_sf, 0.9)
plt.figure(figsize = (8, 8))
plt.plot(range(200000), maskSF) 
plt.plot(range(200000), maskSF2) 
plt.plot(range(200000), maskSF4) 
plt.plot(range(200000), maskSF5)
plt.plot(range(200000), maskSF3) 



                
