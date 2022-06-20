import numpy as np
import cvxpy as cp
# %% number of state
n=3;

# %% number of action
m=3;

# %% transition matrix
# %%T(i,j,k)   transition from state i to state j under action k
# % action 1
T(1,1,1)=1;
T(2,1,1)=0.9;
T(2,2,1)=0.1;
T(3,1,1)=0.2;
T(3,2,1)=0.7;
T(3,3,1)=0.1;

# % action 2
T(1,1,2)=0.9;
T(1,2,2)=0.1;
T(2,2,2)=1;
T(3,2,2)=0.5;
T(3,3,2)=0.5;

# % action 3
T(1,1,3)=0.2;
T(1,2,3)=0.7;
T(1,3,3)=0.1;
T(2,1,3)=0.6;
T(2,3,3)=0.4;
T(3,3,3)=1;