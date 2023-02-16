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
import operator
import random
from random import randrange
import seaborn as sns


def euclideanDistance(dp1, dp2):

    '''
    Calculating the euclidean distance between 2 points
    Inputs taken: iterated each time by taking 2 points from the samples
    Return: Euclidean distance value
    '''

    return np.linalg.norm(dp1 - dp2)


def strategy1Initialization(k, data, centroidsSet):

    '''
    Initial centroids are computed by random selection from data
    Inputs taken: 'k' value (ranging from 2-10), data samples and the centroids set
    Return: Centroids
    '''

    for k in range(k):
        centroidsSet[k] = data[randrange(len(data))]
    return centroidsSet


def KMeans(k, data):

    '''
    Run K means on the given data
    Inputs taken: 
    Return/Output: 
    '''

    centroids = {}

    # Strategy 1 initialization 
    centroids = strategy1Initialization(k,data,centroids)
    

    for r in range(900):

        clusters = {}
        for i in range(k):
            clusters[i] = []

        # Evaluate the euclidean distance
        for points in data:
            distances = [euclideanDistance(points,centroids[centroid]) for centroid in centroids]
            min_dist = distances.index(min(distances))
            clusters[min_dist].append(points)

        prev = dict(centroids)

        # Mean of centroids for updating the centers of clusters
        for c in clusters:
            if clusters[c] != []:
                centroids[c] = np.average(clusters[c], axis = 0)

        converge = True

        # Convergence check
        for centroid in centroids:

            org = prev[centroid]
            curr = centroids[centroid]

            objective = np.sum((curr - org)/org * 100.0)

            if objective > 0.0001:
                converge = False

        if converge:
            break

    return centroids,clusters,objective


def ObjectiveFunc(data):
    objFunc=[]
    for i in range(2,11):

        # Running K means
        centroids,clusters,obj = KMeans(i,data)

        # Visualizing Clusters
        plotClusters(centroids,clusters,i)

        obj=0
        for k in range(i):
            obj+=np.sum((clusters[k]-centroids[k])**2)
        objFunc.append(obj)
    ObjPlot(objFunc)


def ObjPlot(ObjectiveFunc):

    '''
    Plotting the objective function vs number of clusters (k)
    Inputs taken: Objective Function O/P
    Output: Plots
    '''

    K = range(2,11,1)
    plt.plot(K, ObjectiveFunc)
    plt.xlabel('No. of Clusters')
    plt.ylabel('Objective Function')
    plt.title('Objective function vs No. of Clusters (k)')
    plt.show()


def plotClusters(centroids, clusters, i):

    '''
    Visualizing and plotting the clusters as a scatter plot
    Inputs taken: centroids, clusters and k value (ranging from 2-10)
    Output: Plots 
    '''

    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(10)]

    # Generate color palette to differentiate between the clusters
    colorPalette = sns.color_palette(None, len(ys))

    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1], s = 200, marker = "X", c='black')

    for i in clusters:
        color = colorPalette[i]
        for points in clusters[i]:
            plt.scatter(points[0], points[1],color=color,s = 30)

    plt.title("Clusters Visualization for k = " + str(i+1))
    plt.show()


if __name__ == "__main__":

    samples = scipy.io.loadmat("AllSamples.mat")
    data = samples["AllSamples"]
    
    ObjectiveFunc(data)





