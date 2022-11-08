import scipy.sparse as spa
import numpy as np   
from numba import njit
import cvxpy as cp 
# import mosek
import scipy.io
import timeit
import mat73

n = 10
m = 10
alpha1 = 1.0

class BRexist_algo():
  def __init__(self, n, m, T,Num_sample):
    self.n = n
    self.m = m
    self.T = T
    self.Num_sample = Num_sample
   
  def generate_sample(Num_sample):
    Sample = Num_sample
    Sample  = [np.random.rand(n*m,1)-5]*(Num_sample-n*m)
    Sample = np.reshape(Sample,(n*m,Num_sample-n*m))
    Sample = np.concatenate((Sample, np.eye(n*m)),axis = 1)
    return Sample
  @njit(parallel=True, fastmath=True)
  def BRexist(self,n,m,T,ZZ2,notA,alpha1,Chosen_sample):
    # %timeit
    Vert_BRexist = []
    z1 = cp.Variable(n*m)
    z2 = cp.Variable(n*m)
    Z1 = cp.Variable((n*m,5))
    alpha = cp.Variable(len(ZZ2))
    # Chosen_sample = cp.Variable(n*m)
    obj = cp.Minimize(cp.norm(Chosen_sample-z1,2))                  
    cns = [np.ones(n*m).reshape((1,-1)) @ Z1 @ np.ones(5).reshape((-1,1)) == 1,
           Z1 >= 0, 
           Z1@np.ones(5) == z1,
           alpha >= 0,
           cp.sum(alpha) == 1,
           z2 == ZZ2.T@alpha,
           z1[notA]@np.ones(len(notA)) >= alpha1,
           cp.sum(z2) == 1, 
           z2 >= 0]      
          #  np.ones(n*m).reshape((1,-1))@(cp.multiply(T[:,:,2].reshape((n*m,m)),Z1))@np.ones(m).reshape((-1,1)) == z1[2]
          # np.ones(n*m)@z2 
          # np.ones(Num_vert)@alpha == 1
    for i in range(0,n*m-1):
      cns += [np.ones(n*m).reshape((1,-1))@(cp.multiply(T[:,i,:].reshape((n*m,5)),Z1))@np.ones(5).reshape((-1,1)) == z1[i]]
    problem = cp.Problem(obj, cns)
    problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
    Vert_BRexist = z1.value
    return Vert_BRexist
start = timeit.default_timer()

Iter_max = 100
BR_Sample_num = 500
Nmax= 20
numx = numy = 10
a = scipy.io.loadmat('T1010.mat')
transition_mat = a['T']
all_state = [i for i in range(100)]
obs = [21,22,23,26,27,28,29,33,39,42,48,53,59,62,68,71,72,73,74,75,78]
obs[:] = [obs - 1 for obs in obs]
notA = list(set(all_state) - set(obs))
ini_state = [0,1]
alpha1 = 1
alpha2 = 0.8
s = BRexist_algo.generate_sample(BR_Sample_num)
for nnn in range(0,Iter_max-1):
  print("Evaluating", nnn +1 ,"/", Iter_max, "iteration")
  Vert_BRexist_0 = []
  for i in range(BR_Sample_num):
    # Vert_BRexist = [[0 for ii in range(BR_Sample_num)] for jj in range(numx*numy)]
    Chosen_sample = s[:,i]
    z1 = cp.Variable(n*m)
    z2 = cp.Variable(n*m)
    Z1 = cp.Variable((n*m,5))
    # Chosen_sample = cp.Variable(n*m)
    obj = cp.Minimize(cp.norm(Chosen_sample-z1,2))                  
    cns = [np.ones(n*m).reshape((1,-1)) @ Z1 @ np.ones(5).reshape((-1,1)) == 1,
              Z1 >= 0, 
              Z1@np.ones(5) == z1,
              np.ones(n*m)@z2 == 1,
              z2 >= 0,
              z1[notA]@np.ones(len(notA)) >= alpha1,
              z2[ini_state]@np.ones(len(ini_state)) >= alpha2]      
    for k in range(0,n*m-1):
          cns += [np.ones(n*m).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n*m,5)),Z1))@np.ones(5).reshape((-1,1)) == z1[k]]
    problem = cp.Problem(obj, cns)
    problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
    Vert_BRexist_0.append(z1.value)
  # print(z1.value.shape)
  Vert_BRexist_0 = np.array(Vert_BRexist_0)

  Vert_BRexist_union = []
  Vert_BRexist_union.append(Vert_BRexist_0)
  uav5_5 =BRexist_algo(numx,numy,transition_mat,BR_Sample_num)
  
  j = 0
  flag = 1 
  while j <= Nmax and flag == 1:
    # if len(np.nonzero(Vert_BRexist_union[0][ini_state]>0.99)[0])>0.1:
    if len(np.nonzero(np.array([q[ini_state] for q in Vert_BRexist_union]) > 0.99))>0.1:
      item = 1
    else:
      item = 0
    if(np.abs(item - 1) <= 0.1):
        flag = 0
    else:
        zzz1 = []
        sample = BRexist_algo.generate_sample(uav5_5.Num_sample)
        for i in range(BR_Sample_num):
          Chosen_sample = sample[:,i]
          vert_z = uav5_5.BRexist(n,m,transition_mat,Vert_BRexist_0,notA,alpha2,Chosen_sample)
          zzz1.append(vert_z)
        zzz11 = np.array(zzz1)  
        zzz1 = zzz11.conj().transpose()
        zzz1[zzz1<1.000e-8] = 0
        for kkk in range(0,len(zzz1),1):
          zzz1[kkk,:] = zzz1[kkk,:]/np.sum(zzz1[kkk,:])
        Vert_BRexist_union.append(zzz1) 
stop = timeit.default_timer()
print('here')
print('Total Time: ', stop - start)
print('Average Running Time: ', (stop - start)/Iter_max)
print('here') 
  