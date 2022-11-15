import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cvxpy as cp
import scipy.io
import latex


# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/latex'
n= 25
m = 5
alpha1 = 1
alpha2 = 0.9
init_states_idx = 4
pi_ini = np.zeros((n,1))
pi_ini[init_states_idx] = 1
A = [6,11,12,13,18]
notA = [0,1,2,3,4,5,7,8,9,10,14,15,16,17,19,20,21,22,24]
B = [20]
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

#         Z1 >= 0,
#         Z2 >= 0,
#         Z3 >= 0,
#         Z4 >= 0,
#         Z5 >= 0,
#         Z6 >= 0,
#         Z7 >= 0,
#         Z8 >= 0,
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
        np.ones((1,n))@z2 ==1,
        np.ones((1,len(notA)))@Z1[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z2[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z3[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z4[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z5[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z6[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z7[notA]@np.ones((m,1)) >= alpha1,
        np.ones((1,len(notA)))@Z8[notA]@np.ones((m,1)) >= alpha1,
        z2 >= 0,
        z2[notA]@np.ones(len(notA)) >= alpha1,
        z2[B]@np.ones(len(B)) >= alpha2
        ]
# cns = [Z1@np.ones((m,1))==pi_ini,
#         np.ones((1,n))@Z1 >= 0,
#         np.ones((1,n))@Z2 >= 0,
#         np.ones((1,n))@Z3 >= 0,
#         np.ones((1,n))@Z4 >= 0,
#         np.ones((1,n))@Z5 >= 0,
#         np.ones((1,n))@Z6 >= 0,
#         np.ones((1,n))@Z7 >= 0,
#         np.ones((1,n))@Z8 >= 0,
#         np.ones((1,n))@Z1@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z2@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z3@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z4@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z5@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z6@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z7@np.ones((m,1)) ==1,
#         np.ones((1,n))@Z8@np.ones((m,1)) ==1,
#         Z1[notA]@np.ones((m,1)) + Z2[notA]@np.ones((m,1)) 
#         + Z2[notA]@np.ones((m,1))+ Z3[notA]@np.ones((m,1))+ Z4[notA]@np.ones((m,1))+ Z5[notA]@np.ones((m,1))+ Z6[notA]@np.ones((m,1))+ Z7[notA]@np.ones((m,1)) + Z8[notA]@np.ones((m,1)) >= alpha1,
#         np.ones((1,n))@z2 ==1,
#         z2 >= 0,
#         z2[notA]@np.ones(len(notA)) >= alpha1,
#         z2[B]@np.ones(len(B)) >= alpha2
#         ]
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
          
obj = cp.Minimize(cp.norm(z2[A],2)) 
problem = cp.Problem(obj, cns)
problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
pv = []
npv = []
# pv.append(Z1.value@np.ones((m,1)))
# pv.append(Z2.value@np.ones((m,1)))
# pv.append(Z3.value@np.ones((m,1)))
# pv.append(Z4.value@np.ones((m,1)))
# pv.append(Z5.value@np.ones((m,1)))
# pv.append(Z6.value@np.ones((m,1)))
# pv.append(Z7.value@np.ones((m,1)))
# pv.append(Z8.value@np.ones((m,1)))

# pv.append(Z1.value@np.ones((m,1))+2*np.absolute(np.min(Z1.value))+0.1)
# pv.append(Z2.value@np.ones((m,1))+2*np.absolute(np.min(Z2.value))+0.1)
# # pv.append(np.absolute(Z2.value@np.ones((m,1))/np.sum(Z2.value@np.ones((m,1)))))
# pv.append(Z3.value@np.ones((m,1))+2*np.absolute(np.min(Z3.value))+0.1)
# pv.append(Z4.value@np.ones((m,1))+2*np.absolute(np.min(Z4.value))+0.1)
# pv.append(Z5.value@np.ones((m,1))+2*np.absolute(np.min(Z5.value))+0.1)
# pv.append(Z6.value@np.ones((m,1))+2*np.absolute(np.min(Z6.value))+0.1)
# pv.append(Z7.value@np.ones((m,1))+2*np.absolute(np.min(Z7.value))+0.1)
# pv.append(Z8.value@np.ones((m,1))+2*np.absolute(np.min(Z8.value))+0.1)
# print(pv[0])
# print(Z1.value@np.ones((m,1)))
# print(pv)

Z11 = Z1.value@np.ones((m,1))
Z22 = Z2.value@np.ones((m,1))
Z33 = Z3.value@np.ones((m,1))
Z44 = Z4.value@np.ones((m,1))
Z55 = Z5.value@np.ones((m,1))
Z66 = Z6.value@np.ones((m,1))
Z77 = Z7.value@np.ones((m,1))
Z88 = Z8.value@np.ones((m,1))
z22 = z2.value.reshape(-1,1)
# pv = [Z11,Z22,Z33,Z44,Z55,Z66,Z77,Z88]
pv = [Z22,Z33,Z44,Z55,Z66,Z77,Z88,z22]
pv_array = np.array(pv)
# npv.append((Z11-Z11.min())/(Z11.max()-Z11.min()))
# npv.append((Z22-Z22.min())/(Z22.max()-Z22.min()))
# npv.append((Z33-Z33.min())/(Z33.max()-Z33.min()))
# npv.append((Z44-Z44.min())/(Z44.max()-Z44.min()))
# npv.append((Z55-Z55.min())/(Z55.max()-Z55.min()))
# npv.append((Z66-Z66.min())/(Z66.max()-Z66.min()))
# npv.append((Z77-Z77.min())/(Z77.max()-Z77.min()))
# npv.append((Z88-Z88.min())/(Z88.max()-Z88.min()))

# pv_array = np.squeeze(np.array(pv))
# for i in pv_array[i]:
#     array[i] = (pv_array[i] - pv_array[i].min())/(pv_array[i].max() - pv_array[i].min())
# npv_array = preprocessing.MinMaxScaler().fit_transform(pv_array)

# npv_array = np.array(npv)
# subplot1 = npv_array[0].reshape((-5,5)).transpose()
# subplot2 = npv_array[1].reshape((-5,5)).transpose()
# subplot3 = npv_array[2].reshape((-5,5)).transpose()
# subplot4 = npv_array[3].reshape((-5,5)).transpose()
# subplot5 = npv_array[4].reshape((-5,5)).transpose()
# subplot6 = npv_array[5].reshape((-5,5)).transpose()
# subplot7 = npv_array[6].reshape((-5,5)).transpose()
# subplot8 = npv_array[7].reshape((-5,5)).transpose()


pv_array = np.array(pv)
subplot1 = pv_array[0].reshape((5,5)).transpose()
subplot2 = pv_array[1].reshape((5,5)).transpose()
subplot3 = pv_array[2].reshape((5,5)).transpose()
subplot4 = pv_array[3].reshape((5,5)).transpose()
subplot5 = pv_array[4].reshape((5,5)).transpose()
subplot6 = pv_array[5].reshape((5,5)).transpose()
subplot7 = pv_array[6].reshape((5,5)).transpose()
subplot8 = pv_array[7].reshape((5,5)).transpose()
# print(a)

np.shape(a)
#%%
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(2,4)
xlabels = ['1','2','3','4','5']
ylabels = ['5','4','3','2','1']

plt.setp(ax, xticks=np.arange(5), xticklabels=xlabels,
        yticks=np.arange(5), yticklabels=ylabels)


ax[0,0].imshow(subplot1,vmin=0, vmax=1)
ax[0,1].imshow(subplot2,vmin=0, vmax=1)
ax[0,2].imshow(subplot3,vmin=0, vmax=1)
ax[0,3].imshow(subplot4,vmin=0, vmax=1)
ax[1,0].imshow(subplot5,vmin=0, vmax=1)
ax[1,1].imshow(subplot6,vmin=0, vmax=1)
ax[1,2].imshow(subplot7,vmin=0, vmax=1)
a1 = ax[1,3].imshow(subplot8,vmin=0, vmax=1)

# fig.colorbar(a1, ax=ax[:,3])
ax[0,0].set(xlabel=r'(a)$\pi_1$')
ax[0,1].set(xlabel=r'(b) $\pi_2$')
ax[0,2].set(xlabel=r'(c) $\pi_3$')
ax[0,3].set(xlabel=r'(d) $\pi_4$')
ax[1,0].set(xlabel=r'(e) $\pi_5$')
ax[1,1].set(xlabel=r'(f) $\pi_6$')
ax[1,2].set(xlabel=r'(g) $\pi_7$')
ax[1,3].set(xlabel=r'(h) $\pi_8$')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
# fig.colorbar(a1,  cax=cbar_ax)
# arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1)}
# scale = 0.25

# ar =     [['R', 'D', 'L', 'L', 'L'],
#           ['U', 'U', 'L', 'L', 'L'],
#           ['U', 'U', 'L', 'L', 'L'],
#           ['U', 'U', 'L', 'L', 'L'],
#           ['U', 'U', 'L', 'L', 'L']]

Z2n = Z2.value
for i in range (len(Z2.value)):
   Z2n[i] = np.interp(Z2.value[i,:], (Z2.value[i,:].min(), Z2.value[i,:].max()), (-1, +1))

Z3n = Z3.value
for i in range (len(Z3.value)):
   Z3n[i] = np.interp(Z3.value[i,:], (Z3.value[i,:].min(), Z3.value[i,:].max()), (-1, +1))

Z4n = Z4.value
for i in range (len(Z4.value)):
   Z4n[i] = np.interp(Z4.value[i,:], (Z4.value[i,:].min(), Z4.value[i,:].max()), (-1, +1))

Z5n = Z5.value
for i in range (len(Z5.value)):
   Z5n[i] = np.interp(Z5.value[i,:], (Z5.value[i,:].min(), Z5.value[i,:].max()), (-1, +1))
   

Z6n = Z6.value
for i in range (len(Z6.value)):
   Z6n[i] = np.interp(Z6.value[i,:], (Z6.value[i,:].min(), Z6.value[i,:].max()), (-1, +1))

Z7n = Z7.value
for i in range (len(Z7.value)):
   Z7n[i] = np.interp(Z7.value[i,:], (Z7.value[i,:].min(), Z7.value[i,:].max()), (-1, +1))

Z8n = Z8.value
for i in range (len(Z8.value)):
   Z8n[i] = np.interp(Z8.value[i,:], (Z8.value[i,:].min(), Z8.value[i,:].max()), (-1, +1))

z2n = z2.value
z2n = np.interp(z2.value, (z2.value.min(), z2.value.max()), (-1, +1))
   
   
for xx in range (0,5):
    for yy in range (0,5):
        ax[0,0].arrow(xx,yy,Z2n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[0,0].arrow(xx,yy,0,-Z2n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[0,0].arrow(xx,yy,-Z2n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[0,0].arrow(xx,yy,0,-Z2n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[0,0].add_patch( plt.Circle( ( xx, yy ), Z2n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay

for xx in range (0,5):
    for yy in range (0,5):
        ax[0,1].arrow(xx,yy,Z3n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[0,1].arrow(xx,yy,0,-Z3n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[0,1].arrow(xx,yy,-Z3n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[0,1].arrow(xx,yy,0,-Z3n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[0,1].add_patch( plt.Circle( ( xx, yy ), Z3n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay

for xx in range (0,5):
    for yy in range (0,5):
        ax[0,2].arrow(xx,yy,Z4n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[0,2].arrow(xx,yy,0,-Z4n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[0,2].arrow(xx,yy,-Z4n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[0,2].arrow(xx,yy,0,-Z4n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[0,2].add_patch( plt.Circle( ( xx, yy ), Z4n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay
        
for xx in range (0,5):
    for yy in range (0,5):
        ax[0,3].arrow(xx,yy,Z5n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[0,3].arrow(xx,yy,0,-Z5n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[0,3].arrow(xx,yy,-Z5n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[0,3].arrow(xx,yy,0,-Z5n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[0,3].add_patch( plt.Circle( ( xx, yy ), Z5n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay
        
for xx in range (0,5):
    for yy in range (0,5):
        ax[1,0].arrow(xx,yy,Z6n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[1,0].arrow(xx,yy,0,-Z6n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[1,0].arrow(xx,yy,-Z6n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[1,0].arrow(xx,yy,0,-Z6n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[1,0].add_patch( plt.Circle( ( xx, yy ), Z6n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay

for xx in range (0,5):
    for yy in range (0,5):
        ax[1,1].arrow(xx,yy,Z7n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[1,1].arrow(xx,yy,0,-Z7n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[1,1].arrow(xx,yy,-Z7n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[1,1].arrow(xx,yy,0,-Z7n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[1,1].add_patch( plt.Circle( ( xx, yy ), Z7n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay

for xx in range (0,5):
    for yy in range (0,5):
        ax[1,2].arrow(xx,yy,Z8n.reshape(5,5,5)[xx,yy,3]/3,0,head_width=0.03) #L
        ax[1,2].arrow(xx,yy,0,-Z8n.reshape(5,5,5)[xx,yy,0]/3,head_width=0.03)#U
        ax[1,2].arrow(xx,yy,-Z8n.reshape(5,5,5)[xx,yy,2]/3,0,head_width=0.03) #R
        ax[1,2].arrow(xx,yy,0,-Z8n.reshape(5,5,5)[xx,yy,1]/3,head_width=0.03)#D
        ax[1,2].add_patch( plt.Circle( ( xx, yy ), Z8n.reshape(5,5,5)[xx,yy,4]/3, color = 'g') ) #stay
        
ax[1,3].add_patch( plt.Circle( ( 4, 0 ), 1/3, color = 'g') ) #stay
# for r, row in enumerate(ar):
#     for c, cell in enumerate(row):
#         plt.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.1)
plt.show()



# im = ax.imshow(a)
# a = Z1.value@np.ones((m,1))
# print(a[2])
# print(pv[0][2])


#%%
c = Z2.value
for i in range (len(c)):
   c[i] = np.interp(c[i,:], (c[i,:].min(), c[i,:].max()), (0, +1))
    
