import pandas as pd
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import rbf_kernel
from scipy import sparse
from sklearn import svm
from collections import defaultdict
import itertools
from scipy.linalg import eig

def largest_eig(A):
    vals, vectors = eig(A.todense())
    real_indices = [idx for idx, val in enumerate(vals) if not bool(val.imag)]
    vals = [vals[i].real for i in range(len(real_indices))]
    vectors = [vectors[i] for i in range(len(real_indices))]
    max_idx = np.argsort(vals)[-1]
    return np.asarray([vals[max_idx]]), np.asarray([vectors[max_idx]]).T

def get_dependence_matrix(X, gamma): 

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


def get_mod_matrix(X, gamma, comm_nodes=None, B=None):

    if comm_nodes is None:
        comm_nodes = list(range(len(X)))
        return get_dependence_matrix(X,gamma)


    if B is None:
        B = get_dependence_matrix(X,gamma)

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

def improve_dependence(X, gamma, comm_nodes, s, B):
    B_hat_g = get_mod_matrix(X, gamma, comm_nodes, B)
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


def _divide(X, gamma, community_dict, comm_index, B, refine=False):
    comm_nodes = tuple(u for u in community_dict if community_dict[u] == comm_index)
    B_hat_g = get_mod_matrix(X, gamma, comm_nodes, B)
    
    #beta_s, u_s = largest_eig(B_hat_g)
    beta_s, u_s = sparse.linalg.eigs(B_hat_g.toarray(), k=1, which='LR', tol=1E-2)
    u_1 = u_s[:, 0]
    beta_1 = beta_s[0]
    if beta_1 > 0:
        # divisible
        s = sparse.csc_matrix(np.asmatrix([[1 if u_1_i > 0 else -1] for u_1_i in u_1]))
        if refine:
            improve_dependence(X, gamma, comm_nodes, s, B)

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



def partition(X, gamma, refine=True):
    node_name={u:u for u in range(len(X))}
    B=get_dependence_matrix(X, gamma)
    divisible_community = deque([0])
    community_dict = {u: 0 for u in range(len(X))}

    comm_counter = 0
    while len(divisible_community) > 0:

        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(X, gamma, community_dict, comm_index, B, refine)
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

def binary_partition(X, B, gamma, refine=True): #For checking split
    node_name={u:u for u in range(len(X))}
    divisible_community = deque([0])
    community_dict = {u: 0 for u in range(len(X))}

    comm_counter = 0
    while len(divisible_community) == 1:

        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(X, gamma, community_dict, comm_index, B, refine)
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

def outlier_partition(X, B, gamma, refine=True):
    node_name={u:u for u in range(len(X))}
    divisible_community = deque([0])
    community_dict = {u: 0 for u in range(len(X))}

    comm_counter = 0
    while len(divisible_community) > 0:

        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(X, gamma, community_dict, comm_index, B, refine)
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

##Quality
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


def get_Q(X, R, dependence_labels, gamma):

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

# from clx to class(y)
def cly(clx):
    y=np.array([])
    for i in range(len(clx)):
        y=np.append(y,np.repeat(list(clx.keys())[i],len(clx[list(clx.keys())[i]])))
    return y

#from clx to dataframe
def df(clx):
    x=pd.DataFrame() 
    for i in clx.keys():
        x=x.append(clx[i])
    return x


#dictionary construction
def dict_cons(clx, gamma, epsilon):
    rbf_clx=rbf_kernel(clx,gamma=gamma)
    if len(clx)>1:
        init_idx=np.array([],dtype='i')
        init_idx=np.append(init_idx,np.unravel_index(rbf_clx.argmin(),rbf_clx.shape))
        cand_sim=np.vstack([rbf_clx[init_idx[0]],rbf_clx[init_idx[1]]])
        
        if rbf_clx.min()<epsilon:
            while len(clx)-len(init_idx)>0:
                if np.min(cand_sim.max(axis=0))<epsilon:
                    init_idx=np.append(init_idx,np.where(cand_sim.max(axis=0)==np.min(cand_sim.max(axis=0)))[0][0])
                    cand_sim=np.vstack([cand_sim,rbf_clx[init_idx[len(init_idx)-1]]])
                else:
                    break
    
    init_clx=pd.DataFrame()
    for k in init_idx:
        init_clx=init_clx.append(clx.iloc[k:k+1])
    clx=init_clx.copy()
    
    return clx

#KODC
def KODC(X_data, t, tau, update_gap):
    nrow=X_data.shape[0]
    init_row=round(nrow/10) #10% over total data
    
    #Offline Phase
    gamma_list=[1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
    epsilon_list=[0.9,0.99,0.999]
    dependence_list=[]
    for i in gamma_list:
        for j in epsilon_list:
            gamma=i
            epsilon=j
            init_dict=dict_cons(X_data[:init_row], gamma, epsilon)
            dependence_labels=list(partition(init_dict,gamma=gamma).values())
            dependence_list.append(get_Q(X_data[:init_row],init_dict,dependence_labels,gamma=gamma))
    gamma_id=np.argmax(dependence_list)//len(epsilon_list)
    epsilon_id=np.argmax(dependence_list)%len(epsilon_list)
    gamma=gamma_list[gamma_id]
    epsilon=epsilon_list[epsilon_id]
    print('gamma=',gamma, 'epsilon=',epsilon)
    
    init_dict=dict_cons(X_data[:init_row], gamma, epsilon)
    dependence_labels=list(partition(init_dict,gamma=gamma).values())
    
    init_labels=np.array(dependence_labels)
    
    X=X_data[:init_row]
    R=init_dict
    
    A=rbf_kernel(X,R)
    z=np.argmax(A,axis=1)
    xz_pair=dict(zip(list(range(len(X))),z))
    
    for i,j in enumerate(np.unique(init_labels)):
        init_labels=np.where(init_labels==j,i,init_labels)
    
    clx = defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
    clf = defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
    anomaly_clx=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
    anomaly_x=pd.DataFrame()
    
    for i,j in enumerate(np.unique(init_labels)):
        clx[i] = init_dict[init_labels==j]
        clf[i] = svm.OneClassSVM(nu=1/len(clx[i]), kernel="rbf", gamma=gamma) #nu값 setting
        clf[i].fit(clx[i])
    
    for i in list(clx.keys()):
        if len(clx[i])<5:
            anomaly_x=anomaly_x.append(clx[i])
            del(clx[i])
            del(clf[i])
            init_labels=np.where(init_labels==i,-1,init_labels)
    
    for i1,j1 in enumerate(np.unique(list(clx.keys()))):
        clx[i1]=clx.pop(j1)
    for i1,j1 in enumerate(np.unique(list(clf.keys()))):
        clf[i1]=clf.pop(j1)
    
    if -1 in init_labels:
        for i,j in enumerate(np.unique(init_labels)[1:]):
            init_labels=np.where(init_labels==j,i,init_labels)
    else:
        for i,j in enumerate(np.unique(init_labels)):
            init_labels=np.where(init_labels==j,i,init_labels)
            
    predict_y=[]
    for i in range(init_row):
        predict_y=np.append(predict_y,init_labels[xz_pair[i]])


    #Online Phase
    for i in range(init_row,nrow):
    
        clf_decision_value=[]
        for j in list(clf.keys()):
            f_max=clf[j].decision_function(clx[j]).max()
            clf_decision_value.append((f_max-clf[j].decision_function(X_data.iloc[i:i+1]))/f_max)
        g=np.array(list(clx.keys()))[np.argmin(clf_decision_value)]
        score_max=clf[g].decision_function(clx[g]).max()
        score=(score_max-clf[g].decision_function(clx[g]))/score_max
        T=score.mean()+tau*score.std()
        
        if clf_decision_value[g] > T:
            predict_y=np.append(predict_y,-1)
        
        else:
            predict_y=np.append(predict_y,g)
            if np.max(rbf_kernel(X_data.iloc[i:i+1],clx[g],gamma=gamma))<epsilon: #엡실론 값이 높으면 더 많은 데이터를 받는다(오버피팅)
                clx[g]=pd.concat([clx[g],X_data.iloc[i:i+1]],axis=0)
                clf[list(clf.keys())[np.argmax(clf_decision_value)]]=svm.OneClassSVM(nu=1/len(clx[list(clf.keys())[np.argmax(clf_decision_value)]]),kernel="rbf",gamma=gamma)
                clf[list(clf.keys())[np.argmax(clf_decision_value)]].fit(clx[list(clx.keys())[np.argmax(clf_decision_value)]])
    
        if (i%(update_gap)==init_row-1) & (i!=nrow-1):
            print("Check a new cluster")
            if len(anomaly_x)+len(X_data[i-update_gap+1:i+1][predict_y[i-update_gap+1:i+1]==-1])<2:
                anomaly_x=anomaly_x.append(X_data[i-update_gap+1:i+1][predict_y[i-update_gap+1:i+1]==-1])
            else:
                anomaly_x=dict_cons(anomaly_x.append(X_data[i-update_gap+1:i+1][predict_y[i-update_gap+1:i+1]==-1]), gamma, epsilon)
            while len(anomaly_x)>2:
                
                B=get_dependence_matrix(df(clx).append(anomaly_x), gamma=gamma)
                
                indices= np.array(range(len(df(clx)),len(df(clx))+len(anomaly_x)))
                B_g = B[indices, :][:, indices]
                B_hat_g = np.zeros((len(anomaly_x), len(anomaly_x)), dtype=float)
                B_g_rowsum = np.asarray(B_g.sum(axis=1))
    
                for j in range(B_hat_g.shape[0]):
                    for k in range(B_hat_g.shape[0]):
                        if j == k:
                            B_hat_g[j,k] = B_g[j,k] - B_g_rowsum[j]
                        else:
                            B_hat_g[j,k] = B_g[j,k]     
                
                anomaly_c=np.array(list(outlier_partition(anomaly_x, B=B_hat_g, gamma=gamma).values()))
                
                if all(anomaly_c==0):
                    break
                else:
                    anomaly_Q=[]
                    for t,l in enumerate(np.unique(anomaly_c)):
                        anomaly_clx[t]=anomaly_x[anomaly_c==l]
                        anomaly_Q.append(get_Q(X_data[i-update_gap+1:i+1],df(clx).append(anomaly_clx[t]),np.append(cly(clx),np.repeat(max(cly(clx))+1,len(anomaly_clx[t]))).tolist(),gamma=gamma))
                    current_Q=get_Q(X_data[i-update_gap+1:i+1],df(clx),cly(clx).tolist(),gamma=gamma)
                    #print(current_Q)
                    if (current_Q>max(anomaly_Q)) or (len(anomaly_clx[np.argmax(anomaly_Q)])<10):
                        break
                    else:
                        print("New cluster")
                        clx[max(clx.keys())+1]=anomaly_clx[np.argmax(anomaly_Q)]
                        clf[max(clf.keys())+1]=svm.OneClassSVM(nu=1/len(clx[max(clf.keys())+1]),kernel='rbf',gamma=gamma)
                        clf[max(clf.keys())].fit(clx[max(clf.keys())])
                        
                        del(anomaly_clx[np.argmax(anomaly_Q)])
                        anomaly_x=df(anomaly_clx)
                        anomaly_clx=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
    
    
            print("Check Split") #split
            while True:
                cls_sim=[]
                for q in list(clx.keys()):
                    sim_s=rbf_kernel(clx[q],gamma=gamma)            
                    for r in range(len(clx[q])):
                        sim_s[r,r]=0
                    cls_sim.append(sim_s.mean())
                c1=list(clx.keys())[np.argmin(cls_sim)]
    
                B=get_dependence_matrix(df(clx), gamma=gamma)
                    
                indices = np.where(cly(clx)==c1)[0]
                B_g = B[indices, :][:, indices]
                B_hat_g = np.zeros((len(clx[c1]), len(clx[c1])), dtype=float)
                B_g_rowsum = np.asarray(B_g.sum(axis=1))
    
                for j in range(B_hat_g.shape[0]):
                    for k in range(B_hat_g.shape[0]):
                        if j == k:
                            B_hat_g[j,k] = B_g[j,k] - B_g_rowsum[j]
                        else:
                            B_hat_g[j,k] = B_g[j,k]     
                
                split_c=np.array(list(binary_partition(clx[c1], B=B_hat_g, gamma=gamma).values()))
                
                if all(split_c==0):
                    break
                else:
                    current_Q=get_Q(X=X_data[i-update_gap+1:i+1],R=df(clx),dependence_labels=cly(clx).tolist(),gamma=gamma)
                    s1=clx[c1][split_c==np.unique(split_c)[0]]
                    s2=clx[c1][split_c==np.unique(split_c)[1]]
                    clx[max(clx.keys())+1]=s2
                    clx[c1]=s1
                    if (current_Q>get_Q(X_data[i-update_gap+1:i+1],df(clx),cly(clx).tolist(),gamma=gamma)) or (len(s1.T)<10) or (len(s2.T)<10):
                        #print(current_Q) 
                        clx[c1]=s1.append(s2)
                        del(clx[max(clx.keys())])
                        break
                    else:
                        print("Splitting")
                        clf[max(clx.keys())]=svm.OneClassSVM(nu=1/len(clx[max(clx.keys())]),kernel="rbf",gamma=gamma)
                        clf[max(clx.keys())].fit(clx[max(clx.keys())])
                        clf[c1]=svm.OneClassSVM(nu=1/len(clx[c1]),kernel="rbf",gamma=gamma)
                        clf[c1].fit(clx[c1])
    
            print("Check Merger")
            while len(clx)>1:   #merge
                cls_sim=[]
                for o,p in itertools.combinations(clx,2):
                    cls_sim.append(rbf_kernel(clx[o],clx[p]).mean())
                c2,c3=list(itertools.combinations(clx,2))[np.argmax(cls_sim)]
    
                current_Q=get_Q(X_data[i-update_gap+1:i+1],df(clx),cly(clx).tolist(),gamma=gamma)
                m1=clx[min(c2,c3)]
                m2=clx[max(c2,c3)]
                clx[min(c2,c3)]=m1.append(m2)
                del(clx[max(c2,c3)])
                if current_Q>get_Q(X_data[i-update_gap+1:i+1],df(clx),cly(clx).tolist(),gamma=gamma):
                    #print(current_Q)
                    clx[min(c2,c3)]=m1
                    clx[max(c2,c3)]=m2
                    clx=dict(sorted(clx.items()))
                    break
                else:
                    print("Merging")
                    clf[min(c2,c3)]=svm.OneClassSVM(nu=1/len(clx[min(c2,c3)]),kernel="rbf",gamma=gamma)
                    clf[min(c2,c3)].fit(clx[min(c2,c3)])
                    del(clf[max(c2,c3)])
                    
                    for i1,j1 in enumerate(np.unique(list(clx.keys()))):
                        clx[i1]=clx.pop(j1)
                    for i1,j1 in enumerate(np.unique(list(clf.keys()))):
                        clf[i1]=clf.pop(j1)
    return predict_y

#Parameter setting
t=20
tau=6
update_gap=1000

y=KODC(X_data, t, tau, update_gap)


