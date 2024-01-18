import numpy as np
import cvxpy as cp
from numba import jit
from tqdm import tqdm
import networkx as nx

@jit(nopython=True)
def computeKernelMatrix(X, q):
    n = X.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = np.exp(-q * (np.linalg.norm(X[i]-X[j])**2))
    return K

@jit(nopython=True)
def computeKernel(x, X, q):
    """
    For a new x, compute the kernel vector (K(x, X_i))_i of size n

    Args:
        x: a vector of dimension d
        X: a matrix of size n x d
        q: a float
    
    Returns:
        K: a vector of size n
    """
    n = X.shape[0]
    K = np.zeros((n,))
    for i in range(n):
        K[i] = np.exp(-q * (np.linalg.norm(x-X[i])**2))
    return K


def findBeta(K, X, C):
    n = X.shape[0]
    K = cp.psd_wrap(K)
    beta = cp.Variable(n)
    objective = cp.Maximize(cp.sum(beta) - cp.quad_form(beta, K))
    constraints = [beta >= 0, beta <= C, cp.sum(beta) == 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    res = prob.value
    return beta.value

@jit(nopython=True)
def computeAllR(X, K, beta):
    n = X.shape[0]
    r = np.zeros(n)
    for i in range(n):
        r[i] = K[i,i] - 2*np.sum(beta*K[i, :]) + beta.T @ K @ beta
    return r

# np.sum(np.array([beta_val[i]*k[i] for i in range(N)]))

@jit(nopython=True)
def computeR(x, beta_val, quad_form, q, X):
    k = computeKernel(x, X, q)
    N = X.shape[0]
    r = 1 - 2*np.sum(beta_val*k) + quad_form
    return r

@jit(nopython=True)
def sampleSegment(x1, x2, r, quad_form, beta, q, X, n = 10):
    adj = True
    for i in range(n):
        x = x1 + (x2-x1)*i/n
        if (computeR(x, beta, quad_form, q, X) > r):
            adj = False
            return adj
    return adj

#@jit(nopython=True)
def buildAdjacency(X, beta, quad_form, R, C, q):
    n = X.shape[0]
    adj = np.zeros((n,n))
    tmp = np.array(beta<C)*np.array(beta>1e-8)
    support_vectors = np.where(tmp == True)[0]
    tmp = np.array(beta >= C)
    bounded_support_vectors = np.where(tmp == True)[0]

    r = np.mean(R[support_vectors])
    adj = np.zeros((n,n))
    for i in tqdm(range(n)):
        if i not in bounded_support_vectors:
            for j in range(i+1, n):
                if (j not in bounded_support_vectors):
                    if sampleSegment(X[i], X[j], r, quad_form, beta, q, X):
                        adj[i,j] = 1
                        adj[j,i] = 1
    return adj, support_vectors, bounded_support_vectors

def computeClusterIndices(g):
    cluster_indices = []
    for c in nx.connected_components(g):
        cluster_indices.append(list(c))
    return cluster_indices


def computeLabels(cluster_indices, n):
    labels = np.zeros(n)
    for i in range(len(cluster_indices)):
        if len(cluster_indices[i]):
            for j in cluster_indices[i]:
                labels[j] = i
    return labels


def SVC(X, p, q):
    n = X.shape[0]
    C = 1 / (n*p)
    K = computeKernelMatrix(X, q)
    beta = findBeta(K, X, C)
    allR = computeAllR(X, K, beta)
    quad_form = beta.T @ K @ beta
    adj, support_vectors, bounded_support_vectors = buildAdjacency(X, beta, quad_form, allR, C, q)
    g = nx.from_numpy_matrix(adj)
    n_clusters = nx.number_connected_components(g)
    cluster_indices = computeClusterIndices(g)
    labels = computeLabels(cluster_indices, X.shape[0])
    return labels, support_vectors, bounded_support_vectors
