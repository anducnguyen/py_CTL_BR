import scipy.sparse as spa
import numpy as np   
import cvxpy as cp 

class UAVCaseStudy:
  def __init__(self, n, m, T, Num_vert):
    self.n = n
    self.m = m
    self.T = T
    self.Num_vert = Num_vert
  def BRexist(n,m,T,ZZ2,notA,alpha1,Num_sample):
    %timeit
    T = np.array([np.eye(25),np.eye(25),np.eye(25),np.eye(25),np.eye(25)]) # <- trial T
    Num_vert = len(ZZ2)
    Sample  = [np.random.rand(n,1)-5]*(Num_sample-n)
    Sample = np.reshape(Sample,(n,Num_sample-n))
    Sample = np.concatenate((Sample, np.eye(n)),axis = 1)  
    z1 = cp.Variable(n)
    z2 = cp.Variable(n)
    Z1 = cp.Variable(n,m)
    alpha = cp.Parameter(len(ZZ2))
    Chosen_sample = cp.Variable(n)

    obj = cp.Minimize(cvxpy.quad_form(Chosen_sample, spa.eye(n)) - cvxpy.quad_form(z1, spa.eye(n)))                  
    cns = [np.ones(n).reshape((1,-1)) @ Z1 @ np.ones(m).reshape((-1,1)) == 1,
           0 <= Z1, 
           Z1@np.ones(m).reshape((-1,1)) == z1,
           alpha >= np.zeros(Num_vert).reshape((1,-1)),
           np.ones(Num_vert).reshape((-1, 1))@alpha == 1,
           z2 == ZZ2.T@alpha,
           np.sum(z1[notA]) >= alpha1,
           np.ones(n).reshape((-1,1))@z2 ==1,
           z2 >= np.zeros(n).reshape((-1,1))]      

    for i in range(0,n-1):
        cns = [cns, np.ones(n).reshape((1,-1))@((T[:,i,:].reshape(n,m)*Z1))@np.ones(m).reshape((-1,1)) == z1[i]]
    problem = cvxpy.Problem(obj, cns)
    problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True)
    print("Solve time:", problem.solver_stats.solve_time)
  