import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxpy as cp
import scipy.io
import latex
from mpl_toolkits.axes_grid1 import ImageGrid
import os 

# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/latex'
n= 25
m = 5
alpha1 = 1
alpha2 = 0.7
init_states_idx = 20
pi_ini = np.zeros((n,1))
pi_ini[init_states_idx] = 1

notA = [0,1,2,3,4,5,7,8,9,10,14,15,16,17,19,20,21,22,24]
B = [4]
N = 8
a = scipy.io.loadmat('T55_deter.mat')
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
       np.ones((1,n))@Z1@np.ones((m,1)) ==1,
       np.ones((1,n))@Z2@np.ones((m,1)) ==1,
       np.ones((1,n))@Z3@np.ones((m,1)) ==1,
       np.ones((1,n))@Z4@np.ones((m,1)) ==1,
       np.ones((1,n))@Z5@np.ones((m,1)) ==1,
       np.ones((1,n))@Z6@np.ones((m,1)) ==1,
       np.ones((1,n))@Z7@np.ones((m,1)) ==1,
       np.ones((1,n))@Z8@np.ones((m,1)) ==1,
       Z1[notA]@np.ones((m,1)) + Z2[notA]@np.ones((m,1)) 
       + Z2[notA]@np.ones((m,1))+ Z3[notA]@np.ones((m,1))+ Z4[notA]@np.ones((m,1))+ Z5[notA]@np.ones((m,1))+ Z6[notA]@np.ones((m,1))+ Z7[notA]@np.ones((m,1)) + Z8[notA]@np.ones((m,1)) >= alpha1,
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

for i in range(0,n):
           cns += [np.ones(n).reshape((1,-1))@(cp.multiply(transition_mat[:,i,:].reshape((n,m)),Z8))@np.ones(m).reshape((-1,1)) == z2[i]]
          
obj = cp.Minimize(cp.norm(Chosen_sample-z1,2)) 
problem = cp.Problem(obj, cns)
problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
pv = []

# pv.append(Z1.value@np.ones((m,1)))
# pv.append(Z2.value@np.ones((m,1)))
# pv.append(Z3.value@np.ones((m,1)))
# pv.append(Z4.value@np.ones((m,1)))
# pv.append(Z5.value@np.ones((m,1)))
# pv.append(Z6.value@np.ones((m,1)))
# pv.append(Z7.value@np.ones((m,1)))
# pv.append(Z8.value@np.ones((m,1)))


pv.append(Z1.value@np.ones((m,1))+2*np.absolute(np.min(Z1.value))+0.1)
pv.append(Z2.value@np.ones((m,1))+2*np.absolute(np.min(Z2.value))+0.1)
# pv.append(np.absolute(Z2.value@np.ones((m,1))/np.sum(Z2.value@np.ones((m,1)))))
pv.append(Z3.value@np.ones((m,1))+2*np.absolute(np.min(Z3.value))+0.1)
pv.append(Z4.value@np.ones((m,1))+2*np.absolute(np.min(Z4.value))+0.1)
pv.append(Z5.value@np.ones((m,1))+2*np.absolute(np.min(Z5.value))+0.1)
pv.append(Z6.value@np.ones((m,1))+2*np.absolute(np.min(Z6.value))+0.1)
pv.append(Z7.value@np.ones((m,1))+2*np.absolute(np.min(Z7.value))+0.1)
pv.append(Z8.value@np.ones((m,1))+2*np.absolute(np.min(Z8.value))+0.1)
print(pv[0])
print(Z1.value@np.ones((m,1)))
print(pv)



pv_array = np.array(pv)

subplot1 = pv_array[0].reshape((5,5))
subplot2 = pv_array[1].reshape((5,5))
subplot3 = pv_array[2].reshape((5,5))
subplot4 = pv_array[3].reshape((5,5))
subplot5 = pv_array[4].reshape((5,5))
subplot6 = pv_array[5].reshape((5,5))
subplot7 = pv_array[6].reshape((5,5))
subplot8 = pv_array[7].reshape((5,5))
# print(a)

np.shape(a)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(2,4)
xlabels = ['1','2','3','4','5']
ylabels = ['5','4','3','2','1']

plt.setp(ax, xticks=np.arange(5), xticklabels=xlabels,
        yticks=np.arange(5), yticklabels=ylabels)


ax[0,0].imshow(subplot1)
a1 = ax[0,1].imshow(subplot2)
ax[0,2].imshow(subplot3)
ax[0,3].imshow(subplot4)
ax[1,0].imshow(subplot5)
ax[1,1].imshow(subplot6)
ax[1,2].imshow(subplot7)
ax[1,3].imshow(pv_array[7].reshape((5,5)))

# fig.colorbar(a1, ax=ax[:,3])
ax[0,0].set(xlabel=r'(a)$\pi_1$')
ax[0,1].set(xlabel=r'(b) $\pi_2$')
ax[0,2].set(xlabel=r'(c) $\pi_3$')
ax[0,3].set(xlabel=r'(d) $\pi_4$')
ax[1,0].set(xlabel=r'(e) $\pi_5$')
ax[1,1].set(xlabel=r'(f) $\pi_6$')
ax[1,2].set(xlabel=r'(g) $\pi_7$')
ax[1,3].set(xlabel=r'(h) $\pi_8$')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
fig.colorbar(a1,  cax=cbar_ax)
plt.show()



# im = ax.imshow(a)
# a = Z1.value@np.ones((m,1))
# print(a[2])
# print(pv[0][2])
