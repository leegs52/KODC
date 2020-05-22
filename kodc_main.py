import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import svm
from collections import defaultdict
import itertools
import _divide
import utils


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
            init_dict=utils.dict_cons(X_data[:init_row], gamma, epsilon)
            dependence_labels=list(_divide.partition(init_dict, t, gamma=gamma).values())
            dependence_list.append(_divide.get_Q(X_data[:init_row],init_dict,dependence_labels,gamma=gamma, t=t))
    gamma_id=np.argmax(dependence_list)//len(epsilon_list)
    epsilon_id=np.argmax(dependence_list)%len(epsilon_list)
    gamma=gamma_list[gamma_id]
    epsilon=epsilon_list[epsilon_id]
    print('gamma=',gamma, 'epsilon=',epsilon)
    
    init_dict=utils.dict_cons(X_data[:init_row], gamma, epsilon)
    dependence_labels=list(_divide.partition(init_dict, t, gamma=gamma).values())
    
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
        clf[i] = svm.OneClassSVM(nu=1/len(clx[i]), kernel="rbf", gamma=gamma)
        clf[i].fit(clx[i])
    
           
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
            if np.max(rbf_kernel(X_data.iloc[i:i+1],clx[g],gamma=gamma))<epsilon:
                clx[g]=pd.concat([clx[g],X_data.iloc[i:i+1]],axis=0)
                clf[list(clf.keys())[np.argmax(clf_decision_value)]]=svm.OneClassSVM(nu=1/len(clx[list(clf.keys())[np.argmax(clf_decision_value)]]),kernel="rbf",gamma=gamma)
                clf[list(clf.keys())[np.argmax(clf_decision_value)]].fit(clx[list(clx.keys())[np.argmax(clf_decision_value)]])
    
        if (i%(update_gap)==init_row-1) & (i!=nrow-1):
            print("Check a new cluster")
            if len(anomaly_x)+len(X_data[i-update_gap+1:i+1][predict_y[i-update_gap+1:i+1]==-1])<2:
                anomaly_x=anomaly_x.append(X_data[i-update_gap+1:i+1][predict_y[i-update_gap+1:i+1]==-1])
            else:
                anomaly_x=utils.dict_cons(anomaly_x.append(X_data[i-update_gap+1:i+1][predict_y[i-update_gap+1:i+1]==-1]), gamma, epsilon)
            while len(anomaly_x)>2:
                
                B=_divide.get_dependence_matrix(utils.df(clx).append(anomaly_x), gamma=gamma, t=t)
                
                indices= np.array(range(len(utils.df(clx)),len(utils.df(clx))+len(anomaly_x)))
                B_g = B[indices, :][:, indices]
                B_hat_g = np.zeros((len(anomaly_x), len(anomaly_x)), dtype=float)
                B_g_rowsum = np.asarray(B_g.sum(axis=1))
    
                for j in range(B_hat_g.shape[0]):
                    for k in range(B_hat_g.shape[0]):
                        if j == k:
                            B_hat_g[j,k] = B_g[j,k] - B_g_rowsum[j]
                        else:
                            B_hat_g[j,k] = B_g[j,k]     
                
                anomaly_c=np.array(list(_divide.outlier_partition(anomaly_x, B=B_hat_g, gamma=gamma, t=t).values()))
                
                if all(anomaly_c==0):
                    break
                else:
                    anomaly_Q=[]
                    for t,l in enumerate(np.unique(anomaly_c)):
                        anomaly_clx[t]=anomaly_x[anomaly_c==l]
                        anomaly_Q.append(_divide.get_Q(X_data[i-update_gap+1:i+1],utils.df(clx).append(anomaly_clx[t]),np.append(utils.cly(clx),np.repeat(max(utils.cly(clx))+1,len(anomaly_clx[t]))).tolist(),gamma=gamma, t=t))
                    current_Q=_divide.get_Q(X_data[i-update_gap+1:i+1],utils.df(clx),utils.cly(clx).tolist(),gamma=gamma, t=t)
                    #print(current_Q)
                    if current_Q>max(anomaly_Q):
                        break
                    else:
                        print("New cluster")
                        clx[max(clx.keys())+1]=anomaly_clx[np.argmax(anomaly_Q)]
                        clf[max(clf.keys())+1]=svm.OneClassSVM(nu=1/len(clx[max(clf.keys())+1]),kernel='rbf',gamma=gamma)
                        clf[max(clf.keys())].fit(clx[max(clf.keys())])
                        
                        del(anomaly_clx[np.argmax(anomaly_Q)])
                        anomaly_x=utils.df(anomaly_clx)
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
    
                B=_divide.get_dependence_matrix(utils.df(clx), gamma=gamma, t=t)
                    
                indices = np.where(utils.cly(clx)==c1)[0]
                B_g = B[indices, :][:, indices]
                B_hat_g = np.zeros((len(clx[c1]), len(clx[c1])), dtype=float)
                B_g_rowsum = np.asarray(B_g.sum(axis=1))
    
                for j in range(B_hat_g.shape[0]):
                    for k in range(B_hat_g.shape[0]):
                        if j == k:
                            B_hat_g[j,k] = B_g[j,k] - B_g_rowsum[j]
                        else:
                            B_hat_g[j,k] = B_g[j,k]     
                
                split_c=np.array(list(_divide.binary_partition(clx[c1], B=B_hat_g, gamma=gamma, t=t).values()))
                
                if all(split_c==0):
                    break
                else:
                    current_Q=_divide.get_Q(X=X_data[i-update_gap+1:i+1],R=utils.df(clx),dependence_labels=utils.cly(clx).tolist(),gamma=gamma, t=t)
                    s1=clx[c1][split_c==np.unique(split_c)[0]]
                    s2=clx[c1][split_c==np.unique(split_c)[1]]
                    clx[max(clx.keys())+1]=s2
                    clx[c1]=s1
                    if current_Q>_divide.get_Q(X_data[i-update_gap+1:i+1],utils.df(clx),utils.cly(clx).tolist(),gamma=gamma, t=t):
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
    
                current_Q=_divide.get_Q(X_data[i-update_gap+1:i+1],utils.df(clx),utils.cly(clx).tolist(),gamma=gamma, t=t)
                m1=clx[min(c2,c3)]
                m2=clx[max(c2,c3)]
                clx[min(c2,c3)]=m1.append(m2)
                del(clx[max(c2,c3)])
                if current_Q>_divide.get_Q(X_data[i-update_gap+1:i+1],utils.df(clx),utils.cly(clx).tolist(),gamma=gamma, t=t):
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

y=KODC(X_data, t=20, tau=6, update_gap=1000)
