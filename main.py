# Project 2: Unsupervised Learning (K-means) - SML 2022


"""
Created on Sat June 18 2022
@author: Aishwarya Baalaji Rao
"""


# ---------- STRATEGY 1 ----------

import scipy.io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from random import randrange
import operator


def Euclidean_distance(pt1, pt2):

    '''
    Find the Euclidean distance between two points
    Inputs taken: iterated each time by taking 2 points from the samples
    Return/Output: 
    '''

    sum_ = 0

    for i in range(len(pt1)):
            sum_ += (pt1[i] - pt2[i])**2

    d = (sum_)**0.5
    return d


def Initialization_Strategy_1(k,data,centroids_set_1):

    '''
    Calculate the initial centroids according to Strategy 1
    Inputs taken: 
    Return/Output: 
    '''

    for i in range(k):
        centroids_set_1[i] = data[randrange(len(data))]
    return centroids_set_1


def Initialization_Strategy_2(k,data,centroids_set_2):

    '''
    Calculate initial centroids according to Strategy 2
    Inputs taken: k which is the cluster center chosen, data, and the centroids set
    Return/Output: centroids set 
    '''

    chosen = []
    not_chosen = data.tolist()


    for i in range(k):

        if i == 0:

            ch = random.choice(not_chosen)

        else:

            dist = {}
            for j in not_chosen:
                sum_d = 0
                for l in chosen:
                    sum_d += Euclidean_distance(j,l)
                dist[tuple(j)] = sum_d/len(chosen)

            sorted_dist = sorted(dist.items(), key=operator.itemgetter(1))
            ch = sorted_dist[-1][0]

        chosen.append(np.array(ch))
        not_chosen.remove(list(ch))
   
    k_count=0
    for c in chosen:
        centroids_set_2[k_count] = c
        k_count += 1

    return centroids_set_2


def K_means(k,data):

    '''
    Run K means on the given data
    Inputs taken: 
    Return/Output: 
    '''

    centroids = {}

    # Initialization
    centroids = Initialization_Strategy_1(k,data,centroids)
    

    for i in range(500):

        clusters = {}
        for i in range(k):
            clusters[i] = []

        # Find Euclidean Distance
        for points in data:
            distances = [Euclidean_distance(points,centroids[centroid]) for centroid in centroids]
            min_dist = distances.index(min(distances))
            clusters[min_dist].append(points)

        prev = dict(centroids)

        # Avg over centroids
        for i in clusters:
            centroids[i] = np.average(clusters[i], axis = 0)

        converge = True

        # Check convergence
        for centroid in centroids:

            org = prev[centroid]
            curr = centroids[centroid]

            objective = np.sum((curr - org)/org * 100.0)

            if objective > 0.0001:
                converge = False

        if converge:
            break

    return centroids,clusters,objective


def Objective_func():
    Obj_fun=[]
    for i in range(2,11):

        # Running K means
        centroids,clusters,obj = K_means(i,data)

        # Visualizing Clusters
        Visualize_Clusters(centroids,clusters,i)

        obj=0
        for k in range(i):
            obj+=np.sum((clusters[k]-centroids[k])**2)
        Obj_fun.append(obj)
    Objective_Plot(Obj_fun)


def Objective_Plot(Objective_fun):

    '''
    Plot Objective function vs K
    Inputs taken: 
    Output: 
    '''

    K_x=range(2,11,1)
    plt.plot(K_x,Objective_fun)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Value of Objective Function')
    plt.title('Objective function vs No. of Clusters (k)')
    plt.show()


def Visualize_Clusters(centroids,clusters,i):

    '''
    Function to visualize clusters
    Inputs taken: 
    Return/Output: 
    '''

    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(10)]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1], s = 200, marker = "X", c='black')

    for i in clusters:
        color = colors[i]
        for points in clusters[i]:
            plt.scatter(points[0], points[1],color=color,s = 30)

    plt.title("Plot for k = "+str(i))
    plt.show()


if __name__ == "__main__":

    raw_data = scipy.io.loadmat("AllSamples.mat")
    data = raw_data["AllSamples"]
    
    # Calling Objective function
    Objective_func()









