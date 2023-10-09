import numpy as np

def edge2mat(link, num_node):
    """
    convert edge links to adjacency matrix

    :param link: list of edge links
    :num_node: number of nodes to construct matrix
    :returns: adjacency matrix
    """
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    """
    normalise a directed graph

    :param A: Adjacency matrix
    :return: 
    """
    # Diagonal Degree of A + I?
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    """
    builds the spatial graph

    :param num_node: number of nodes in the graph
    :param self_link: self link edges
    :param inward: inward edges
    :param outward: outward edges
    :return: complete spatial graph
    """
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out)) # stack all the 
    return A