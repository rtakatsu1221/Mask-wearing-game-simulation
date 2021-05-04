#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:24:55 2021

@author: rikuyatakatsu
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
  

G_sf = nx.to_undirected(nx.scale_free_graph(n = 10000))
G_6 = nx.watts_strogatz_graph(n = 10000, k = 6, p = 0.5)
G_gr_6 = nx.watts_strogatz_graph(n = 10000, k = 6, p = 0)
G_r_6 = nx.watts_strogatz_graph(n = 10000, k = 6, p = 1)

colors = ['blue', 'red']

plt.figure(figsize = (8, 8))
plt.title("Number of degrees")
plt.hist([G_gr_4.degree[i] for i in G_gr_4.nodes()], label="Grid", alpha = 0.6, color="coral")
plt.hist([G_4.degree[i] for i in G_4.nodes()], bins = 20, label="Small World", alpha = 0.6, color="green")
plt.hist([G_r_4.degree[i] for i in G_r_4.nodes()], bins = 20, label = "Random", alpha = 0.6, color = "pink")
plt.legend()
nx.draw(G_gr, node_size = 50)

def initGame(graph, thres, metathres):
    for node in graph.nodes():
        if (np.random.uniform(0, 1) < thres):
            strategy = 0 # mask
        else:
            strategy = 1 # no mask
            
# =============================================================================
               
        if (np.random.uniform(0, 1) < metathres):
            metastrategy = 0 # immitate the best
        else:
            metastrategy = 1 # generalized tit-for-tat
            
# =============================================================================

            
        graph.nodes[node]['strategy'] = strategy
        graph.nodes[node]['metastrategy'] = metastrategy
        graph.nodes[node]['payoff'] = 0
        graph.nodes[node]['infected'] = 0
        graph.nodes[node]['symptom'] = 0 # symptomatic agents don't play
        graph.nodes[node]['err'] = np.random.uniform(0, 1) * 1 - 1/2
        if (np.random.uniform(0, 1) < 0.001): 
            graph.nodes[node]['infected'] = 14
            symptom = np.random.uniform(0, 1)
            if (symptom < 2/3): 
                graph.nodes[node]['strategy'] = 0
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
            if (symptom < 1/6):
                graph.nodes[node]['symptom'] = 1
                graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50

def getPayoff(graph, node, C): #plays games with neighbors, updates payoff and returns best responce
    mask = 0
    nonMask = 0
    graph.nodes[node]['payoff'] = 0
    
    for v in list(graph.adj[node]):
        if (graph.nodes[v]['symptom'] == 0):
            if (graph.nodes[v]['strategy'] == 0):
                mask = mask + 1
                if (graph.nodes[node]['strategy'] == 0):
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1
                    if (graph.nodes[node]['infected'] == 0 and 
                        graph.nodes[v]['infected'] > 0 and
                        np.random.uniform(0, 1) < 0.05): # change the param here
                        graph.nodes[node]['infected'] = 14
                        symptom = np.random.uniform(0, 1)
                        if (symptom < 2/3):
                            graph.nodes[node]['strategy'] = 0
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
                        if (symptom < 1/6):
                            graph.nodes[node]['symptom'] = 1 
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
                else:
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] + 1 + C
                    if (graph.nodes[node]['infected'] == 0 and 
                        graph.nodes[v]['infected'] > 0 and
                        np.random.uniform(0, 1) < 0.1): # change the param here
                        graph.nodes[node]['infected'] = 14
                        symptom = np.random.uniform(0, 1)
                        if (symptom < 2/3):
                            graph.nodes[node]['strategy'] = 0
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
                        if (symptom < 1/6):
                            graph.nodes[node]['symptom'] = 1 
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
            else:
                nonMask = nonMask + 1
                if (graph.nodes[node]['strategy'] == 0):
                    graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - C
                    if (graph.nodes[node]['infected'] == 0 and 
                        graph.nodes[v]['infected'] > 0 and
                        np.random.uniform(0, 1) < 0.25): # change the param here
                        graph.nodes[node]['infected'] = 14
                        symptom = np.random.uniform(0, 1)
                        if (symptom < 2/3):
                            graph.nodes[node]['strategy'] = 0
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
                        if (symptom < 1/6):
                            graph.nodes[node]['symptom'] = 1 
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
                else:
                    if (graph.nodes[node]['infected'] == 0 and 
                        graph.nodes[v]['infected'] > 0 and
                        graph.nodes[v]['symptom'] == 0 and
                        np.random.uniform(0, 1) < 0.5): # change the param here
                        graph.nodes[node]['infected'] = 14
                        symptom = np.random.uniform(0, 1)
                        if (symptom < 2/3): 
                            graph.nodes[node]['strategy'] = 0
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
                        if (symptom < 1/6):
                            graph.nodes[node]['symptom'] = 1
                            graph.nodes[node]['payoff'] = graph.nodes[node]['payoff'] - 50
# ===returns the strategy played by the majority===========================================================
    if (mask > nonMask):
        return 0
    elif (nonMask > mask):
        return 1
    if (np.random.uniform(0, 1) < 0.5):
        return 0
    return 1


def simulate(niter, C, graph, thres, metathres):
    initGame(graph, thres, metathres) 
    mask = list()
    totalInfected = list()
    
    for i in range(niter):
        maskCount = 0
        infectedCount = 0
        for node in graph.nodes():
            if (graph.nodes[node]['infected'] > 0):
                graph.nodes[node]['infected'] = graph.nodes[node]['infected'] - 1 
                infectedCount = infectedCount + 1
            else:
                if (graph.nodes[node]['strategy'] == 0):
                    maskCount = maskCount + 1
                    
        mask.append(maskCount/len(graph.nodes()))
        totalInfected.append(infectedCount/len(graph.nodes()))

        for node in graph.nodes():
            if (graph.nodes[node]['symptom'] == 0): # only play if susceptible/asymptotic
                br = getPayoff(graph, node, C)
# ===used when metastrategy = 1 is generalized Tit-for-Tat=====================
#                if (graph.nodes[node]['metastrategy'] == 1):
#                    graph.nodes[node]['strategy'] = br
# =============================================================================
        
        for node in graph.nodes(): 
            if (graph.nodes[node]['symptom'] == 0 and graph.nodes[node]['metastrategy'] == 0):
                maxNode = node
                maxPayoff = graph.nodes[node]['payoff']
                maxStrategy = graph.nodes[node]['strategy']
                for v in list(graph.adj[node]):
                    if (graph.nodes[v]['symptom'] == 0 and graph.nodes[v]['payoff'] > maxPayoff):
                        maxPayoff = graph.nodes[v]['payoff']
                        maxStrategy = graph.nodes[v]['strategy']
                        maxNode = v
                if (maxPayoff > graph.nodes[node]['payoff']):
                    if (graph.nodes[node]['metastrategy'] == 0 and np.random.uniform(0, 1) < 1/(1 + math.exp((maxPayoff - graph.nodes[node]["payoff"])/0.1))):
                        graph.nodes[node]['strategy'] = maxStrategy
                        
# ===used when metastrategy = 1 is imitation with error term===================
#                if (graph.nodes[node]['metastrategy'] == 1 and maxPayoff > graph.nodes[node]['payoff'] + graph.nodes[node]['err']): 
#                    graph.nodes[node]['strategy'] = maxStrategy    
# =============================================================================
# ===used when metastrategy = 1 is deterministic imitation=====================
                if (graph.nodes[node]['metastrategy'] == 1):
                    graph.nodes[node]['strategy'] = maxStrategy
# =============================================================================
                    
        graph.nodes[np.random.randint(len(graph.nodes()))]['strategy'] = abs(graph.nodes[np.random.randint(len(graph.nodes()))]['strategy'] - 1)
        randInfected = np.random.randint(len(graph.nodes()))
        graph.nodes[randInfected]['infected'] = 14
        symptom = np.random.uniform(0, 1)
        if (symptom < 2/3): 
            graph.nodes[randInfected]['strategy'] = 0
            graph.nodes[randInfected]['payoff'] = graph.nodes[randInfected]['payoff'] - 50
        if (symptom < 1/6):
            graph.nodes[randInfected]['symptom'] = 1
            graph.nodes[randInfected]['payoff'] = graph.nodes[randInfected]['payoff'] - 50
        
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

maskSW = simulate(1000, 0.5, G_4, 0.5, 0)
# maskSF = simulate(1000, 0.5, G_sf, 0.5, 0)
maskGR = simulate(1000, 0.5, G_gr_4, 0.5, 0)
maskR = simulate(1000, 0.5, G_r_4, 0.5, 0)

plt.figure(figsize = (8, 8))
plt.title("People wearing masks (%)")

plt.plot(range(1000), [100*i for i in asympDet4GR[0]], color="coral", label="Grid")
plt.plot(range(1000), [100*i for i in asympDet4SW[0]], color="green", label="Small World")
# plt.plot(range(1000), [100*i for i in maskSF[0]], color="skyblue", label="Scale Free")
plt.plot(range(1000), [100*i for i in asympDet4R[0]], color="pink", label="Random")
plt.legend()
plt.ylim(0,100)

plt.figure(figsize = (8, 8))
plt.title("Infected (%)") 

plt.plot(range(1000), [100*i for i in asympDet4GR[1]], color="coral", label="Grid")
plt.plot(range(1000), [100*i for i in asympDet4SW[1]], color="green", label="Small World")
# plt.plot(range(1000), [100*i for i in maskSF[1]], color="skyblue", label="Scale Free")
plt.plot(range(1000), [100*i for i in asympDet4R[1]], color="pink", label="Random")
plt.legend()
plt.ylim(0,80)

mildSympTFT4GR #= maskGR
mildSympTFT4SW #= maskSW
mildSympTFT4R #= maskR

mildSympExp4GR #= maskGR
mildSympExp4SW #= maskSW
mildSympExp4R #= maskR

mildSympErr4GR #= maskGR
mildSympErr4SW #= maskSW
mildSympErr4R #= maskR

mildSympDet4GR #= maskGR
mildSympDet4SW #= maskSW
mildSympDet4R #= maskR

asympTFT4GR #= maskGR
asympTFT4SW #= maskSW
asympTFT4R #= maskR

asympExp4GR #= maskGR
asympExp4SW #= maskSW
asympExp4R #= maskR

asympErr4GR #= maskGR
asympErr4SW #= maskSW
asympErr4R #= maskR

asympDet4GR #= maskGR
asympDet4SW #= maskSW
asympDet4R #= maskR

asympExp6GR #= maskGR
asympExp6SW #= maskSW
asympExp6R #= maskR


    
                