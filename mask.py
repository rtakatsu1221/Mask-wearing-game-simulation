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
G_r = nx.watts_strogatz_graph(n = 10000, k = 4, p = 1)

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
        graph.nodes[node]['symptom'] = 0 # symptomatic agents don't play
        if (np.random.uniform(0, 1) < 0.001): 
            graph.nodes[node]['infected'] = 14
            if (np.random.uniform(0, 1) < (1/6)):
                graph.nodes[node]['symptom'] = 1 
                graph.nodes[node]['payoff'] = -100 
                graph.nodes[node]['strategy'] = 0

def getPayoff(graph, node, C, cumulative):
    if (cumulative == 0):
        graph.nodes[node]['payoff'] = 0
    for v in list(graph.adj[node]):
        if (graph.nodes[node]['strategy'] == 0):
            if (graph.nodes[v]['strategy'] == 0):
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    graph.nodes[v]['symptom'] == 0 and
                    np.random.uniform(0, 1) < 0.05): # change the param here
                    graph.nodes[node]['infected'] = 14
                    if (np.random.uniform(0, 1) < (1/6)):
                        graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
                        graph.nodes[node]['symptom'] = 1 
                        graph.nodes[node]['strategy'] = 0
            else:
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - C
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    graph.nodes[v]['symptom'] == 0 and
                    np.random.uniform(0, 1) < 0.25): # change the param here
                    graph.nodes[node]['infected'] = 14
                    if (np.random.uniform(0, 1) < (1/6)):
                        graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
                        graph.nodes[node]['symptom'] = 1 
                        graph.nodes[node]['strategy'] = 0
        else:
            if (graph.nodes[v]['strategy'] == 0):
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1 + C
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    graph.nodes[v]['symptom'] == 0 and
                    np.random.uniform(0, 1) < 0.10): # change the param here
                    graph.nodes[node]['infected'] = 14
                    if (np.random.uniform(0, 1) < (1/6)):
                        graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
                        graph.nodes[node]['symptom'] = 1 
                        graph.nodes[node]['strategy'] = 0
            else:
                if (graph.nodes[node]['infected'] == 0 and 
                    graph.nodes[v]['infected'] > 0 and
                    graph.nodes[v]['symptom'] == 0 and
                    np.random.uniform(0, 1) < 0.5): # change the param here
                    graph.nodes[node]['infected'] = 14
                    if (np.random.uniform(0, 1) < (1/6)):
                        graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 100
                        graph.nodes[node]['symptom'] = 1 
                        graph.nodes[node]['strategy'] = 0
    
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
            if (graph.nodes[node]['symptom'] == 0):
                getPayoff(graph, node, C, cumulative)
        
        if (i % 7 == 0):
            for node in graph.nodes(): # can we make this update less frequent ?
                if (graph.nodes[node]['symptom'] == 0):
                    maxNode = node
                    maxPayoff = graph.nodes[node]['payoff']
                    maxStrategy = graph.nodes[node]['strategy']
                    for v in list(graph.adj[node]):
                        if (graph.nodes[v]['symptom'] == 0 and graph.nodes[v]['payoff'] > maxPayoff):
                            maxPayoff = graph.nodes[v]['payoff']
                            maxStrategy = graph.nodes[v]['strategy']
                            maxNode = v
                    if (maxPayoff > graph.nodes[node]['payoff']):
                        if (np.random.uniform(0, 1) < (graph.nodes[maxNode]['payoff'] - graph.nodes[node]['payoff'])/(max(len(list(graph.adj[node])), len(list(graph.adj[maxNode])))*(1+C))):
                            graph.nodes[node]['strategy'] = maxStrategy
                       
            for node in graph.nodes():
                graph.nodes[node]['payoff'] = 0
                       
            
        graph.nodes[np.random.randint(len(graph.nodes()))]['strategy'] = abs(graph.nodes[np.random.randint(len(graph.nodes()))]['strategy'] - 1)
        graph.nodes[np.random.randint(len(graph.nodes()))]['infected'] = 14
        
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
        
nx.draw(G_gr, node_size = 50, node_color=[colors[G.nodes[node]['strategy']] for node in G.nodes])


maskSW = simulate(10000, 0.5, G, 0.5, 1)
maskSF = simulate(10000, 0.5, G_sf, 0.5, 1)
maskGR = simulate(10000, 0.5, G_gr, 0.5, 1)
maskR = simulate(10000, 0.5, G_r, 0.5, 1)

plt.figure(figsize = (8, 8))
plt.title("People wearing masks (%)")
plt.plot(range(10000), [100*i for i in maskSW[0]], color="green", label="Small World")
plt.plot(range(10000), [100*i for i in maskSF[0]], color="skyblue", label="Scale Free")
plt.plot(range(10000), [100*i for i in maskGR[0]], color="coral", label="Grid")
plt.plot(range(10000), [100*i for i in maskR[0]], color="pink", label="Random")
plt.legend()
plt.ylim(0,100)

plt.figure(figsize = (8, 8))
plt.title("Infected (%)")
plt.plot(range(10000), [100*i for i in maskSW[1]], color="green", label="Small World")
plt.plot(range(10000), [100*i for i in maskSF[1]], color="skyblue", label="Scale Free")
plt.plot(range(10000), [100*i for i in maskGR[1]], color="coral", label="Grid")
plt.plot(range(10000), [100*i for i in maskR[1]], color="pink", label="Random")
plt.legend()
plt.ylim(0,100)

maskSWsinglesim = maskSW
maskSFsinglesim = maskSF
maskGRsinglesim = maskGR
maskRsinglesim = maskR

maskSWsingledetsim = maskSW
maskSFsingledetsim = maskSF
maskGRsingledetsim = maskGR
maskRsingledetsim = maskR

maskSW7daysdetsim = maskSW
maskSF7daysdetsim = maskSF
maskGR7daysdetsim = maskGR
maskR7daysdetsim = maskR




                