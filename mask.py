#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:24:55 2021

@author: rikuyatakatsu
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
  
G = nx.watts_strogatz_graph(n = 10000, k = 4, p = 0.5)
G_sf = nx.to_undirected(nx.scale_free_graph(n = 10000))
G_gr = nx.watts_strogatz_graph(n = 10000, k = 4, p = 0)

colors = ['blue', 'red']

plt.figure(figsize = (12, 12))
nx.draw(G_gr, node_size = 50, node_color=[colors[G_gr.nodes[node]['strategy']] for node in G_gr.nodes])

def initGame(graph, thres):
    for node in graph.nodes():
        if (np.random.uniform(0, 1) < thres):
            strategy = 0 # mask
        else:
            strategy = 1 # no mask
        graph.nodes[node]['strategy'] = strategy
        graph.nodes[node]['payoff'] = 0
        graph.nodes[node]['infected'] = 0
        if (np.random.uniform(0, 1) < 0.001): 
            graph.nodes[node]['infected'] = 14
            

def getPayoff(graph, node, C, cumulative):
    if (cumulative == 0):
        graph.nodes[node]['payoff'] = 0
    for v in list(graph.adj[node]):
        if (graph.nodes[node]['strategy'] == 0):
            if (graph.nodes[v]['strategy'] == 0):
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    np.random.uniform(0, 1) < 0.05): # change the param here
                    graph.nodes[node]['infected'] = 14
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
            else:
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - C
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    np.random.uniform(0, 1) < 0.25): # change the param here
                    graph.nodes[node]['infected'] = 14
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
        else:
            if (graph.nodes[v]['strategy'] == 0):
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1 + C
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    np.random.uniform(0, 1) < 0.10): # change the param here
                    graph.nodes[node]['infected'] = 14
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
            else:
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    np.random.uniform(0, 1) < 0.5): # change the param here
                    graph.nodes[node]['infected'] = 14
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
    
def simulate(niter, C, graph, thres, cumulative):
    initGame(graph, thres) 
    mask = list()
    totalInfected = list()
    
    for i in range(niter):
        maskCount = 0
        infectedCount = 0
        for node in graph.nodes():
            if (graph.nodes[node]['strategy'] == 0):
                maskCount = maskCount + 1
            if (graph.nodes[node]['infected'] > 0):
                graph.nodes[node]['infected'] = graph.nodes[node]['infected'] - 1 
                infectedCount = infectedCount + 1
        mask.append(maskCount/len(graph.nodes()))
        totalInfected.append(infectedCount/len(graph.nodes()))

        for node in graph.nodes():
            getPayoff(graph, node, C, cumulative)
        
        for node in graph.nodes():
            maxPayoff = graph.nodes[node]['payoff']
            maxStrat = graph.nodes[node]['strategy']
            for v in list(graph.adj[node]):
                if (graph.nodes[v]['payoff'] > maxPayoff):
                    maxPayoff = graph.nodes[v]['payoff']
                    maxStrat = graph.nodes[v]['strategy']
            if (maxPayoff > graph.nodes[node]['payoff']):
                graph.nodes[node]['strategy'] = maxStrat # deterministically change
        
        graph.nodes[np.random.randint(len(graph.nodes()))]['strategy'] = abs(graph.nodes[np.random.randint(len(graph.nodes()))]['strategy'] - 1)
            
        
# ==choosing single vertex to update===========================================
#         u = np.random.randint(len(graph.nodes()))
#         
#         if (np.random.uniform(0, 1) < 0): ## random change
#             graph.nodes[u]['strategy'] = abs(graph.nodes[u]['strategy'] - 1)
#         else :
#             getPayoff(graph, u, C, cumulative)
#             v = np.random.randint(len(list(graph.adj[u])))
#             getPayoff(graph, list(graph.adj[u])[v], C, cumulative)
#             if (graph.nodes[v]['payoff'] > graph.nodes[u]['payoff']):
#                 if (np.random.uniform(0, 1) < 
#                     (graph.nodes[v]['payoff'] - graph.nodes[u]['payoff'])/(max(len(list(graph.adj[u])), len(list(graph.adj[v])))*(1+C))):
#                     graph.nodes[u]['strategy'] = graph.nodes[v]['strategy']
# =============================================================================
        
    return (mask, totalInfected) 
        
maskSW = simulate(50000, 0.7, G, 0.25, 0)
maskSW2 = simulate(50000, 0.7, G, 0.5, 0)
maskSW4 = simulate(50000, 0.7, G, 0.75, 0)
maskSW5 = simulate(50000, 0.7, G, 0.85, 0)
maskSW3 = simulate(50000, 0.7, G, 0.9, 0)
plt.figure(figsize = (8, 8))
plt.title("SINGLE PAYOFF")
plt.plot(range(50000), maskSW) 
plt.plot(range(50000), maskSW2) 
plt.plot(range(50000), maskSW4) 
plt.plot(range(50000), maskSW5)
plt.plot(range(50000), maskSW3) 

maskSW = simulate(50000, 0.5, G, 0.25, 1)
maskSW2 = simulate(50000, 0.5, G, 0.5, 1)
maskSW4 = simulate(50000, 0.5, G, 0.75,1)
maskSW5 = simulate(50000, 0.5, G, 0.85, 1)
maskSW3 = simulate(50000, 0.5, G, 0.9, 1)
plt.figure(figsize = (8, 8))
plt.title("CUMULATIVE")
plt.plot(range(50000), maskSW) 
plt.plot(range(50000), maskSW2) 
plt.plot(range(50000), maskSW4) 
plt.plot(range(50000), maskSW5)
plt.plot(range(50000), maskSW3) 
plt.figure(figsize = (12, 12))
nx.draw(G, node_size = 50, node_color=[colors[G.nodes[node]['strategy']] for node in G.nodes])


maskSF = simulate(200000, 1.9, 0.7, G_sf, 0.25)
maskSF2 = simulate(50000, 0.5, G_sf, 0.5, 1)
maskSF4 = simulate(50000, 0.5, G_sf, 0.75, 1)
maskSF5 = simulate(200000, 1.9, 0.7, G_sf, 0.85)
maskSF3 = simulate(200000, 1.9, 0.7, G_sf, 0.9)
plt.figure(figsize = (8, 8))
plt.plot(range(200000), maskSF) 
plt.plot(range(200000), maskSF2) 
plt.plot(range(200000), maskSF4) 
plt.plot(range(200000), maskSF5)
plt.plot(range(200000), maskSF3) 

maskSW = simulate(1000, 0.5, G, 0.5, 1)
maskSF = simulate(1000, 0.5, G_sf, 0.5, 1)
maskGR = simulate(1000, 0.5, G_gr, 0.5, 1)

plt.figure(figsize = (8, 8))
plt.title("Percentage of people wearing masks")
plt.plot(range(1000), maskSW[0], color="green")
plt.plot(range(1000), maskSF[0], color="skyblue")
plt.plot(range(1000), maskGR[0], color="coral")

plt.figure(figsize = (8, 8))
plt.title("Percentage of total infected")
plt.plot(range(1000), maskSW[1], color="green", label="Small World")
plt.plot(range(1000), maskSF[1], color="skyblue", label="Scale Free")
plt.plot(range(1000), maskGR[1], color="coral", label="Random Graph")



                