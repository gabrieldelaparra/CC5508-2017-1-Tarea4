#import os
#from scipy import ndimage as ndi
# , morphology, filters, exposure, measure, segmentation, color, img_as_float
from skimage import io
#import matplotlib.pylab as plt
#import matplotlib
import numpy as np

#from ipywidgets import interact, widgets, HBox
#from IPython import display
import math
#from random import random, randint

#from disjointSet import Set
#from unionFind import UnionFind

# Ajustar el tamaño de las imágenes:
#matplotlib.rcParams['figure.figsize'] = (14,12)


class PixelNode:
    def __init__(self, image, x, y):
        self.x = x
        self.y = y
        self.r = int(image[x][y][0])
        self.g = int(image[x][y][0])
        self.b = int(image[x][y][2])

    def __str__(self):
        return "px: " + str(self.x) + " py: " + str(self.y)

class PixelEdge:
    def __init__(self, pixelNode1, pixelNode2):
        self.node1 = pixelNode1
        self.node2 = pixelNode2
        self.weight = 0

        def pixelDistance(self):
            self. weigth = math.sqrt(math.pow(pixelNode1.r - pixelNode2.r, 2)
                                + math.pow(pixelNode1.g - pixelNode2.g, 2)
                                + math.pow(pixelNode1.b - pixelNode2.b, 2))

    def __str__(self):
        return "Node1: " + str(self.node1) + " Node2: " + str(self.node2) + " Weight: " + str(self.weight)

class PixelGraph:
    def __init__(self, image):
        self.nodes = []
        self.edges = []
        getGraph(image)

        def getGraph(self, image):
            nodes = []
            edges = []
            rows, cols, _ = image.shape
            for row in range(rows):
                for col in range(cols):
                    node = PixelNode(image, row, col)
                    nodes.append(node)
                    if row + 1 < rows:
                        edges.append(
                            PixelEdge(node, PixelNode(image, row + 1, col)))

                    if col + 1 < cols:
                        edges.append(
                            PixelEdge(node, PixelNode(image, row, col + 1)))

                    if row + 1 < rows and col + 1 < cols:
                        edges.append(
                            PixelEdge(node, PixelNode(image, row + 1, col + 1)))

                    if row - 1 >= 0 and col + 1 < cols:
                        edges.append(
                            PixelEdge(node, PixelNode(image, row - 1, col + 1)))
        self.nodes = nodes
        self.edges = edges

###############################################################################

# class UnionFindNode:
#     def __init__(self, i, k):
#         self.parent = i
#         self.rank = 0
#         self.threshold = k # k / innerDifference
#         self.size = 1

#     def Find(self, node):
#         if(self.parent[node] != node):
#             self.parent[node] = self.Find(self.parent[node])
#         return self.parent[node]

#     def Union(self, x, y):
#         xRoot = self.Find(x)
#         yRoot = self.Find(y)
#         if(xRoot == yRoot):
#             return
#         if self.rank < self.rank[yRoot]:
#             self.parent[xRoot] = yRoot
#         elif self.rank[xRoot] > self.rank[yRoot]:
#             self.parent[yRoot] = xRoot
#         else:
#             self.parent[yRoot] = xRoot
#             self.rank[xRoot]+=1
#
#     def join(self, x, y):
#         if self.elements[x].rank > self.elements[y].rank:
#             self.elements[y].parent = x
#             self.elements[x].size += self.elements[y].size
#         else:
#             self.elements[x].parent = y
#             self.elements[y].size += self.elements[x].size
#             if self.elements[x].rank == self.elements[y].rank:
#                 self.elements[y].rank += 1

# def MakeSet(x):
#      x.parent = x
#      x.rank   = 0

# def Union(x, y):
#      xRoot = Find(x)
#      yRoot = Find(y)
#      if xRoot.rank > yRoot.rank:
#          yRoot.parent = xRoot
#      elif xRoot.rank < yRoot.rank:
#          xRoot.parent = yRoot
#      elif xRoot != yRoot: # Unless x and y are already in same set, merge them
#          yRoot.parent = xRoot
#          xRoot.rank = xRoot.rank + 1

# def Find(x):
#      if x.parent == x:
#         return x
#      else:
#         x.parent = Find(x.parent)
#         return x.parent


class DisjointSet:
    def __init__(self, n, k):
        self.parent = [x for x in range(n)]
        self.rank = [0 for x in range(n)]
        self.size = [1 for x in range(n)]
        self.threshold = [k for x in range(n)]
        self.thresholdConstant = k

    def Find(self, node):
        if(self.parent[node] != node):
            self.parent[node] = self.Find(self.parent[node])
        return self.parent[node]

    def Union(self, node1, node2, weigthThreshold):
        node1Root = self.Find(node1)
        node2Root = self.Find(node2)

        if(node1Root != node2Root):
            if (weigthThreshold <= node1Root.threshold) and (weigthThreshold <= node2Root.threshold):

                if self.rank[node1Root] < self.rank[node2Root]:
                    self.parent[node1Root] = node2Root
                    self.size[node1Root] += self.size[node2Root]
                    self.threshold[node1Root] = weigthThreshold + \
                        self.thresholdConstant / self.size[node1Root]

                elif self.rank[node1Root] > self.rank[node2Root]:
                    self.parent[node2Root] = node1Root
                    self.size[node2Root] += self.size[node1Root]
                    self.threshold[node2Root] = weigthThreshold + \
                        self.thresholdConstant / self.size[node2Root]

                else:
                    self.parent[node2Root] = node1Root
                    self.threshold[node2Root] = weigthThreshold + \
                        self.thresholdConstant / self.size[node2Root]
                    self.rank[node1Root] += 1


# def thresholdFunction(size, k):
#     return k / size

def segmentGraph(graph, k):
    nodes, edges = graph

    edges = sorted(edges, lambda e: e.weight)

    disjointSet = DisjointSet(len(nodes), k)

    for i in range(len(edges)):
        disjointSet.Union(nodes.index(edges[i].node1), nodes.index(
            edges[i].node2), edges[i].weight)

        # node1Root = disjointSet.Find(edges[i].node1)
        # node2Root = disjointSet.Find(edges[i].node2)

        # if (node1 != node2):
        #         # Check if they need to be joined (and do it)
        #     if (edges[i].weight <= node1Root.threshold) and (edges[i].weight <= node2Root.threshold):
        #         disjointSet.Union(node1Root, node2Root, edges[i].weigth)
        #         vertex1 = disjointSet.Find(node1Root)
        #          node1Root.threshold = edges[i].weight + thresholdFunction(disjointSet.size(vertex1), k)

    # subtree = [TreeNode(i) for i in range(len(nodes))]

    # # sets =  [str(Find(x)) for x in nodes]

    # # Make a disjoint-set forest
    # sets = Set(len(nodes))

    # # Initialize thresholds (size=1)
    # threshold = np.zeros(len(nodes))
    # for i in range(len(nodes)):
    #     threshold[i] = thresholdFunction(1, k)

    # # For each edge (non-decreasing order)
    # for i in range(len(edges)):
    #     # Vertices connected by this edge
    #     vertex1 = sets.find(edges[i].vertex1)
    #     vertex2 = sets.find(edges[i].vertex2)
    #     # If they belong to different components
    #     if (vertex1 != vertex2):
    #         # Check if they need to be joined (and do it)
    #         if (edges[i].weight <= threshold[vertex1]) and (edges[i].weight <= threshold[vertex2]):
    #             sets.join(vertex1, vertex2)
    #             vertex1 = sets.find(vertex1)
    #             threshold[vertex1] = edges[i].weight + thresholdFunction(sets.size(vertex1), k)

    return disjointSet


print("start")
img = io.imread('photos-1-17_05.jpg')
nodes, edges = getGraph(img)
print(len(edges))
disjointSet = segmentGraph(0.7)
print(len(disjointSet))
