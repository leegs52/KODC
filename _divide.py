import pandas as pd
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import rbf_kernel
from scipy import sparse
from sklearn import svm
from collections import defaultdict
import itertools
from scipy.linalg import eig



def get_dependence_matrix(X, gamma, t):

    W=rbf_kernel(X,gamma=gamma)
    for i in range(len(X)):
        for j in range(len(X)):
            if i==j:
                W[i,j]=(W[i].sum()-W[i,j])/(len(X)-1)
    P=np.eye(len(X))
    for i in range(len(X)):
        for j in range(len(X)):
            P[i,j]=W[i,j]/W[i].sum()
    
    P_t=matrix_product(P,t)
    
    B_t=np.eye(len(X))
    for i in range(len(X)):
        B_t[i,i]=P_t[:,i].sum()/len(X)
        
    B=np.dot(P_t,np.linalg.pinv(B_t))-1  #d_0=1
    
    return B


def get_mod_matrix(X, t, gamma, comm_nodes=None, B=None):

    if comm_nodes is None:
        comm_nodes = list(range(len(X)))
        return get_dependence_matrix(X,gamma, t)


    if B is None:
        B = get_dependence_matrix(X,gamma, t)

    indices = [list(range(len(X))).index(u) for u in comm_nodes]

    B_g = B[indices, :][:, indices]
    B_hat_g = np.zeros((len(comm_nodes), len(comm_nodes)), dtype=float)
    B_g_rowsum = np.asarray(B_g.sum(axis=1))

    for i in range(B_hat_g.shape[0]):
        for j in range(B_hat_g.shape[0]):
            if i == j:
                B_hat_g[i,j] = B_g[i,j] - B_g_rowsum[i]

            else:
                B_hat_g[i,j] = B_g[i,j]

    return sparse.csc_matrix(B_hat_g)



def get_delta_Q(B_hat_g, s):
    delta_Q = (s.T.dot(B_hat_g)).dot(s)
    return delta_Q[0,0]

def improve_dependence(X, gamma, t, comm_nodes, s, B):
    B_hat_g = get_mod_matrix(X, t, gamma, comm_nodes, B)
    while True:
        unmoved = list(comm_nodes)
        node_indices=np.array([], dtype=int)
        node_improvement = np.array([], dtype=float)

        while len(unmoved)>0:
            Q0=get_delta_Q(B_hat_g,s)
            scores=np.zeros(len(unmoved))

            for k_index in range(scores.size):
                k = comm_nodes.index(unmoved[k_index])
                s[k, 0] = -s[k, 0]
                scores[k_index] = get_delta_Q(B_hat_g, s) - Q0
                s[k, 0] = -s[k, 0]

            _j=np.argmax(scores)
            j=comm_nodes.index(unmoved[_j])
            s[j,0]=-s[j,0]
            node_indices=np.append(node_indices,j)

            if node_improvement.size<1:
                node_improvement=np.append(node_improvement,scores[_j])

            else:
                node_improvement=np.append(node_improvement, node_improvement[-1]+scores[_j])

            unmoved.pop(_j)

        max_index=np.argmax(node_improvement)

        for i in range(max_index+1, len(comm_nodes)):
            j=node_indices[i]
            s[j,0]=-s[j,0]

        if max_index==len(comm_nodes)-1:
            delta_dependence=0

        else:
            delta_dependence=node_improvement[max_index]

        if delta_dependence<=0:
            break


def _divide(X, gamma, t, community_dict, comm_index, B, refine=False):
    comm_nodes = tuple(u for u in community_dict if community_dict[u] == comm_index)
    B_hat_g = get_mod_matrix(X, t, gamma, comm_nodes, B)
    
    #beta_s, u_s = largest_eig(B_hat_g)
    beta_s, u_s = sparse.linalg.eigs(B_hat_g.toarray(), k=1, which='LR', tol=1E-2)
    u_1 = u_s[:, 0]
    beta_1 = beta_s[0]
    if beta_1 > 0:
        # divisible
        s = sparse.csc_matrix(np.asmatrix([[1 if u_1_i > 0 else -1] for u_1_i in u_1]))
        if refine:
            improve_dependence(X, gamma, t, comm_nodes, s, B)

        delta_dependence = get_delta_Q(B_hat_g, s)
        if delta_dependence > 0:
            g1_nodes = np.array([comm_nodes[i] for i in range(u_1.shape[0]) if s[i,0] > 0])
            if len(g1_nodes) == len(comm_nodes) or len(g1_nodes) == 0:
                # indivisble, return None
                return None, None

            # divisible, return node list for one of the groups
            return g1_nodes, comm_nodes
    # indivisble, return None
    return None, None 



def partition(X, t, gamma, refine=True):
    node_name={u:u for u in range(len(X))}
    B=get_dependence_matrix(X, gamma, t)
    divisible_community = deque([0])
    community_dict = {u: 0 for u in range(len(X))}

    comm_counter = 0
    while len(divisible_community) > 0:

        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(X, gamma, t, community_dict, comm_index, B, refine)
        if g1_nodes is None:
            continue

        g1=g1_nodes
        g2=set(comm_nodes).difference(set(g1_nodes))       

        comm_counter += 1
        divisible_community.append(comm_counter)
     
        for u in g1:
            community_dict[u] = comm_counter

        comm_counter += 1
        divisible_community.append(comm_counter)

        for u in g2:
            community_dict[u] = comm_counter
          
    return {node_name[u]: community_dict[u] for u in range(len(X))}

def binary_partition(X, B, gamma, t, refine=True): #For checking split
    node_name={u:u for u in range(len(X))}
    divisible_community = deque([0])
    community_dict = {u: 0 for u in range(len(X))}

    comm_counter = 0
    while len(divisible_community) == 1:

        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(X, gamma, t, community_dict, comm_index, B, refine)
        if g1_nodes is None:
            continue

        g1=g1_nodes
        g2=set(comm_nodes).difference(set(g1_nodes))       

        comm_counter += 1
        divisible_community.append(comm_counter)
     
        for u in g1:
            community_dict[u] = comm_counter

        comm_counter += 1
        divisible_community.append(comm_counter)

        for u in g2:
            community_dict[u] = comm_counter
          
    return {node_name[u]: community_dict[u] for u in range(len(X))}

def outlier_partition(X, B, gamma, t, refine=True):
    node_name={u:u for u in range(len(X))}
    divisible_community = deque([0])
    community_dict = {u: 0 for u in range(len(X))}

    comm_counter = 0
    while len(divisible_community) > 0:

        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(X, gamma, t, community_dict, comm_index, B, refine)
        if g1_nodes is None:
            continue

        g1=g1_nodes
        g2=set(comm_nodes).difference(set(g1_nodes))       

        comm_counter += 1
        divisible_community.append(comm_counter)
     
        for u in g1:
            community_dict[u] = comm_counter

        comm_counter += 1
        divisible_community.append(comm_counter)

        for u in g2:
            community_dict[u] = comm_counter
          
    return {node_name[u]: community_dict[u] for u in range(len(X))}
    


def largest_eig(A):
    vals, vectors = eig(A.todense())
    real_indices = [idx for idx, val in enumerate(vals) if not bool(val.imag)]
    vals = [vals[i].real for i in range(len(real_indices))]
    vectors = [vectors[i] for i in range(len(real_indices))]
    max_idx = np.argsort(vals)[-1]
    return np.asarray([vals[max_idx]]), np.asarray([vectors[max_idx]]).T

def matrix_product(X,t):
    P=X
    for i in range(t-1):
        P=np.dot(P,X)
    return P

def get_transition_matrix(X, t, gamma): 

    W=rbf_kernel(X,gamma=gamma)
    for i in range(len(X)):
        for j in range(len(X)):
            if i==j:
                W[i,j]=(W[i].sum()-W[i,j])/(len(X)-1)
    P=np.eye(len(X))
    for i in range(len(X)):
        for j in range(len(X)):
            P[i,j]=W[i,j]/W[i].sum()
    
    P_t=matrix_product(P,t)
    
    return P_t


def get_Q(X, R, dependence_labels, gamma, t):

    for i in R.index:
        if i in X.index:
            X=X.drop(i,0)
    X=X.append(R)
            
    tran_X=get_transition_matrix(X,t=1,gamma=gamma)
    tran_R=get_transition_matrix(R,t=t,gamma=gamma) #t조정 가능
    
    A=rbf_kernel(X,R)
    z=np.argmax(A,axis=1)
    xz_pair=dict(zip(list(range(len(X))),z))

    C=np.eye(len(X))
    for i in range(len(X)):
        for j in range(len(X)):
            C[i,j]=tran_X[i,len(X)-len(R)+xz_pair[i]]*tran_R[xz_pair[i],xz_pair[j]]*tran_X[len(X)-len(R)+xz_pair[j],j]

    B_t=np.eye(len(X))
    for i in range(len(X)):
        B_t[i,i]=C[:,i].sum()/len(X)
          
    D=np.dot(C,np.linalg.pinv(B_t))-1

    
    #D=(D-1)/D.sum()
    
    R_labels=pd.DataFrame([list(range(len(R))),dependence_labels]).T
    R_labels.columns=['R','labels']
    Q=0
    X_labels=[]
    for i in range(len(X)):
        X_labels.append(int(R_labels['labels'][R_labels['R']==xz_pair[i]]))
    
    for i in range(len(X)):
        for j in range(len(X)):
            if (X_labels[i]==X_labels[j]) & (X_labels[i]!=-1) & (X_labels[j]!=-1):
                Q=Q+D[i,j]
    return Q
        