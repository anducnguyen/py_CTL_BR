import cvxpy as cp
import numpy as np
import mosek 
import pyny3d as pyny

# Problem data.


# number of state
n=3;

# number of action
m=3;

# m = 30
# n = 20
np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m)
zz1 = cp.Variable(n)
# print(zz1)
cns_xx1 = []
cns_xx1 = [cns_xx1, np.ones(n)@zz1 ==1]
cns_xx1 = [cns_xx1, zz1>=0]
for i in range(1,n,1):
    cns_xx1 = [cns_xx1, -0.1<=zz1(i)-1/3<=0.1]




# Construct the problem.
x = cp.Variable(n)
xobjective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve(solver=cp.MOSEK)
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)
print (cp.installed_solvers())