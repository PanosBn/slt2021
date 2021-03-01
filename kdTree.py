import numpy as np
import copy
import pandas as pd
import math
from metrics import euclidean_distance, minkowski_distance

class KdTree(list):
    """
    this tree recursively finds the median of a certain dimension of X and divides the nodes in 
    groups smaller and greater than the median 
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.dimensions = len(X[0])
        self.root = None

        # initialize the input X and y as KnNode objects as these can have parents and children
        init_nodes = []
        for x,y in zip(X,y):
            init_nodes.append(KdNode(x,y))
        self.nodes = copy.copy(init_nodes) 
        # copy.copy is used to 'shallow copy' nodes and not create new instances as this would
        # make the tree not useable as children and parents are not stored correctly

        def build_tree(nodes, dimension, side=None):
            """
            recursive function
            param dimension: int
            """
            def find_nearest(array, value):
                """
                helper function that returns the index of the element that is closest to the median
                in a list
                """
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx

            median = np.median([node.X[dimension] for node in nodes])
            median_node_index = find_nearest([node.X[dimension] for node in nodes], median)
            median_node = copy.copy(nodes[median_node_index]) # we use this node to further build the subtree

            nodes_right, nodes_left = [], []

            # shallow copy nodes
            for node in nodes:
                if node != median_node:
                    node.parent = median_node # every nodes except the median node gets the median node as parent
                    if node.X[dimension] >= median: # here we divide the children in groups smaller and greater than the median
                        nodes_right.append(copy.copy(node)) 
                    else:
                        nodes_left.append(copy.copy(node)) # again we shallow copy

            # to debug
            #print(f"dimension: {dimension}, median: {median}, median_node: {median_node.y}")

            if dimension != 0: # so if it is not our initial call and we havent got a root node
                if side == "left":
                    median_node.parent.left_children = [median_node] # so link the child of the parent to current node
                elif side == "right":
                    median_node.parent.right_children = [median_node]
            else:
                self.root = median_node # useful to know where to start when searching

            dimension += 1 # to further divide

            if(dimension >= self.dimensions - 1): # base case so here we could have multiple children/leaves
                median_node.left_children = nodes # ???this is the parent
            else: # recursive case
                if nodes_left: # if not empty
                    build_tree(nodes_left, dimension, side="left")
                elif nodes_right:
                    build_tree(nodes_right, dimension, side="right")

        build_tree(init_nodes, 0) # initial call


    def search(self, search_X, distance_func=euclidean_distance):
        """
        in this function we calculate the distance between the search_X and each node we pass
        we then check if our search_X at a dimension is greater or smaller than the node we are at
        and then recursively check call the function of that nodes child (which is only 1 in the 
        recursive calls, and could be multiple children/leaves in the base case).

        when using this in the looc we also return the distance between the same node (=0).
        Thus then we should increase the n_neighbors with 1 and subtract the first in our result.
        """
        distances_labels = []
        distance = (distance_func(search_X, self.root.X), self.root.y) 
        distances_labels.append(distance)

        def search_recursion(current_node, search_X, dimension):

            distance = (distance_func(search_X, current_node.X), current_node.y)
            distances_labels.append(distance) 

            if (search_X[dimension] >= current_node.X[dimension]): # check if its greater than the median
                # search right children
                children = copy.copy(current_node.right_children)
            else:
                # search left children
                children = copy.copy(current_node.left_children)

            if children: # if we even have children
                if len(children) > 1: # if it has multiple children it can only be the base case 
                    for child in children:
                        # in the base case we brute force calculate every distance between the children
                        distance = (distance_func(search_X, child.X), child.y)
                        distances_labels.append(distance) 
                    return distances_labels

                elif len(children) == 1: # recursive case, children is here just one child
                    dimension += 1 # update dimension
                    search_recursion(*children, search_X, dimension) # unpack children as it is a list of one child
        
        search_recursion(self.root, search_X, 0)

        labels = sorted(distances_labels, key=lambda x:x[0])

        labels = [x[1] for x in labels if x[0] != 0.0] # so if it is not itself included
        return labels

    
    def __str__(self):
        return f"tree of length {len(self.nodes)}"

class KdNode:
    def __init__(self, X, y, parent=None, left_children=None, right_children=None):
        self.parent = parent
        self.left_children = left_children
        self.right_children = right_children
        self.X = X
        self.y = y
    
    def __eq__(self, other): 
        return (self.X==other.X).all()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"KdNode ({self.X}, {self.y})"
    