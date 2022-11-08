import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxpy as cp
import scipy.io
import latex
import os 

# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/latex'
n= 25
m = 5
alpha1 = 0.85
alpha2 = 0.8
init_states_idx = 20
pi_ini = np.zeros((n,1))
pi_ini[init_states_idx] = 1

notA = [0,1,2,3,4,6,7,8,9,10,14,15,16,17,19,20,21,22,24]
B = [4]
N = 8
a = scipy.io.loadmat('T55_noisy.mat')
transition_mat = a['T']

z1 = cp.Variable(n)
z2 = cp.Variable(n)
Z1 = cp.Variable((n,m))
Z2 = cp.Variable((n,m))
Z3 = cp.Variable((n,m))
Z4 = cp.Variable((n,m))
Z5 = cp.Variable((n,m))
Z6 = cp.Variable((n,m))
Z7 = cp.Variable((n,m))
Z8 = cp.Variable((n,m))
Z9 = cp.Variable((n,m))
Z10 = cp.Variable((n,m))
Z11 = cp.Variable((n,m))
Z12 = cp.Variable((n,m))
Z13 = cp.Variable((n,m))
Z14 = cp.Variable((n,m))
Z15 = cp.Variable((n,m))

Chosen_sample = cp.Variable((n))
cns = [Z1@np.ones((m,1))==pi_ini,
       np.ones((1,n))@Z1 >= 0,
       np.ones((1,n))@Z2 >= 0,
       np.ones((1,n))@Z3 >= 0,
       np.ones((1,n))@Z4 >= 0,
       np.ones((1,n))@Z5 >= 0,
       np.ones((1,n))@Z6 >= 0,
       np.ones((1,n))@Z7 >= 0,
       np.ones((1,n))@Z8 >= 0,
       np.ones((1,n))@Z9 >= 0,
       np.ones((1,n))@Z10 >= 0,
       np.ones((1,n))@Z11 >= 0,
       np.ones((1,n))@Z12 >= 0,
       np.ones((1,n))@Z13 >= 0,
       np.ones((1,n))@Z14 >= 0,
       np.ones((1,n))@Z15 >= 0,
       np.ones((1,n))@Z1@np.ones((m,1)) ==1,
       np.ones((1,n))@Z2@np.ones((m,1)) ==1,
       np.ones((1,n))@Z3@np.ones((m,1)) ==1,
       np.ones((1,n))@Z4@np.ones((m,1)) ==1,
       np.ones((1,n))@Z5@np.ones((m,1)) ==1,
       np.ones((1,n))@Z6@np.ones((m,1)) ==1,
       np.ones((1,n))@Z7@np.ones((m,1)) ==1,
       np.ones((1,n))@Z8@np.ones((m,1)) ==1,
       np.ones((1,n))@Z9@np.ones((m,1)) ==1,
       np.ones((1,n))@Z10@np.ones((m,1)) ==1,
       np.ones((1,n))@Z11@np.ones((m,1)) ==1,
       np.ones((1,n))@Z12@np.ones((m,1)) ==1,
       np.ones((1,n))@Z13@np.ones((m,1)) ==1,
       np.ones((1,n))@Z14@np.ones((m,1)) ==1,
       np.ones((1,n))@Z15@np.ones((m,1)) ==1,
       Z1[notA]@np.ones((m,1)) + Z2[notA]@np.ones((m,1)) 
       + Z3[notA]@np.ones((m,1))+ Z4[notA]@np.ones((m,1))
       + Z5[notA]@np.ones((m,1))+ Z6[notA]@np.ones((m,1))
       + Z7[notA]@np.ones((m,1)) + Z8[notA]@np.ones((m,1)) 
       + Z9[notA]@np.ones((m,1)) + Z10[notA]@np.ones((m,1)) 
       + Z11[notA]@np.ones((m,1)) + Z12[notA]@np.ones((m,1)) 
       + Z13[notA]@np.ones((m,1)) + Z14[notA]@np.ones((m,1)) 
       + Z15[notA]@np.ones((m,1)) >= alpha1,
       np.ones((1,n))@z2 ==1,
       z2 >= 0,
       z2[notA]@np.ones(len(notA)) >= alpha1,
       z2[B]@np.ones(len(B)) >= alpha2
       ]
for k in range(0,n):
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z1))@np.ones(m).reshape((-1,1)) == Z2[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z2))@np.ones(m).reshape((-1,1)) == Z3[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z3))@np.ones(m).reshape((-1,1)) == Z4[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z4))@np.ones(m).reshape((-1,1)) == Z5[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z5))@np.ones(m).reshape((-1,1)) == Z6[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z6))@np.ones(m).reshape((-1,1)) == Z7[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z7))@np.ones(m).reshape((-1,1)) == Z8[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z8))@np.ones(m).reshape((-1,1)) == Z9[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z9))@np.ones(m).reshape((-1,1)) == Z10[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z10))@np.ones(m).reshape((-1,1)) == Z11[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z11))@np.ones(m).reshape((-1,1)) == Z12[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z12))@np.ones(m).reshape((-1,1)) == Z13[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z13))@np.ones(m).reshape((-1,1)) == Z14[k,:]@np.ones(m).reshape((-1,1))]
          cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,k,:].reshape((n,m)),Z14))@np.ones(m).reshape((-1,1)) == Z15[k,:]@np.ones(m).reshape((-1,1))]


for i in range(0,n):
           cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,i,:].reshape((n,m)),Z15))@np.ones(m).reshape((-1,1)) == z2[i]]
          
obj = cp.Minimize(cp.norm(Chosen_sample-z1,2)) 
problem = cp.Problem(obj, cns)
problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
pv = []

pv.append(Z1.value@np.ones((m,1))+2*np.absolute(np.min(Z1.value))+0.1)
pv.append(Z2.value@np.ones((m,1))+2*np.absolute(np.min(Z2.value))+0.1)
pv.append(Z3.value@np.ones((m,1))+2*np.absolute(np.min(Z3.value))+0.1)
pv.append(Z4.value@np.ones((m,1))+2*np.absolute(np.min(Z4.value))+0.1)
pv.append(Z5.value@np.ones((m,1))+2*np.absolute(np.min(Z5.value))+0.1)
pv.append(Z6.value@np.ones((m,1))+2*np.absolute(np.min(Z6.value))+0.1)
pv.append(Z7.value@np.ones((m,1))+2*np.absolute(np.min(Z7.value))+0.1)
pv.append(Z8.value@np.ones((m,1))+2*np.absolute(np.min(Z8.value))+0.1)
pv.append(Z9.value@np.ones((m,1))+2*np.absolute(np.min(Z9.value))+0.1)
pv.append(Z10.value@np.ones((m,1))+2*np.absolute(np.min(Z10.value))+0.1)
pv.append(Z11.value@np.ones((m,1))+2*np.absolute(np.min(Z11.value))+0.1)
pv.append(Z12.value@np.ones((m,1))+2*np.absolute(np.min(Z12.value))+0.1)
pv.append(Z13.value@np.ones((m,1))+2*np.absolute(np.min(Z13.value))+0.1)
pv.append(Z14.value@np.ones((m,1))+2*np.absolute(np.min(Z14.value))+0.1)
pv.append(Z15.value@np.ones((m,1))+2*np.absolute(np.min(Z15.value))+0.1)

pv_array = np.array(pv)

subplot1 = pv_array[0].reshape((5,5))
subplot2 = pv_array[1].reshape((5,5))
subplot3 = pv_array[2].reshape((5,5))
subplot4 = pv_array[3].reshape((5,5))
subplot5 = pv_array[4].reshape((5,5))
subplot6 = pv_array[5].reshape((5,5))
subplot7 = pv_array[6].reshape((5,5))
subplot8 = pv_array[7].reshape((5,5))
subplot9 = pv_array[8].reshape((5,5))
subplot10 = pv_array[9].reshape((5,5))
subplot11 = pv_array[10].reshape((5,5))
subplot12 = pv_array[11].reshape((5,5))
subplot13 = pv_array[12].reshape((5,5))
subplot14 = pv_array[13].reshape((5,5))
subplot15 = pv_array[14].reshape((5,5))
# print(a)

np.shape(a)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(3,5,figsize=(15, 15))
xlabels = ['1','2','3','4','5']
ylabels = ['5','4','3','2','1']

plt.setp(ax, xticks=np.arange(5), xticklabels=xlabels,
        yticks=np.arange(5), yticklabels=ylabels)

ax[0,0].imshow(subplot1)
ax[0,1].imshow(subplot2)
ax[0,2].imshow(subplot3)
ax[0,3].imshow(subplot4)
a1 = ax[0,4].imshow(subplot5)
ax[1,0].imshow(subplot6)
ax[1,1].imshow(subplot7)
ax[1,2].imshow(subplot8)
ax[1,3].imshow(subplot9)
ax[1,4].imshow(subplot10)
ax[2,0].imshow(subplot11)
ax[2,1].imshow(subplot12)
ax[2,2].imshow(subplot13)
ax[2,3].imshow(subplot14)
ax[2,4].imshow(subplot15)
# fig.colorbar(a1, ax=ax[:,3])

ax[0,0].set(xlabel=r'(a)$\pi_1$')
ax[0,1].set(xlabel=r'(b) $\pi_2$')
ax[0,2].set(xlabel=r'(c) $\pi_3$')
ax[0,3].set(xlabel=r'(d) $\pi_4$')
ax[0,4].set(xlabel=r'(e) $\pi_5$')
ax[1,0].set(xlabel=r'(g)$\pi_6$')
ax[1,1].set(xlabel=r'(h) $\pi_7$')
ax[1,2].set(xlabel=r'(i) $\pi_8$')
ax[1,3].set(xlabel=r'(j) $\pi_9$')
ax[1,4].set(xlabel=r'(k) $\pi_{10}$')
ax[2,0].set(xlabel=r'(l)$\pi_{11}$')
ax[2,1].set(xlabel=r'(m) $\pi_{12}$')
ax[2,2].set(xlabel=r'(n) $\pi_{13}$')
ax[2,3].set(xlabel=r'(p) $\pi_{14}$')
ax[2,4].set(xlabel=r'(q) $\pi_{15}$')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
fig.colorbar(a1, cax=cbar_ax)
plt.show()



# im = ax.imshow(a)
# a = Z1.value@np.ones((m,1))
# print(a[2])
# print(pv[0][2])
