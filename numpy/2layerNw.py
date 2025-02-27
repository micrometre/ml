import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([  [0,0,1], [0,1,1], [1,0,1], [1,1,1] ])

# output dataset           
y = np.array([[0,0,1,1]]).T
 
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
for iter in range(10000):
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1,True)
    syn0 += np.dot(l0.T,l1_delta)
print("Output After Training:")


print(X)
