# Exercise 1 of Andrew Ng's Machine Learning class on Coursera
# Only the first problem. The multi-variable case is similar
# Feb 15, 2015. by Xingliang Ma

# Things learnt: 
# 1. Python is NOT Matlab!!!
# 2. Vector is different from matrix (reshape vectors to matrix for matrix operations)
# 3. c_[a, b] vs. [a, b]
# 4. genfromtxt() to import text file (It took me so long to figure it out. Can't believe it)

from numpy import *
import matplotlib.pyplot as plt
import os

# set the directory
path = "/Users/horse/Desktop/Coursera/MachineLearning/mlclass-ex1-008/mlclass-ex1/python"
os.chdir(path)

#%% 1. warmUpExercise
A = eye(5)
print "\n"*2, "\n An 5x5 identity matrix is: ", "\n", A

#%% Get data and plot
data1 = genfromtxt("ex1data1.txt", delimiter=',')
#data2 = genfromtxt("ex1data2.txt", delimiter=',')  # multi-variable is the same
n1 = len(data1)

X = data1[:, 0]  # features
y = data1[:, 1]
y = y.reshape(n1,1) # Be careful of the vectors!!!  Just make it a 2-D array.

print "\n", "A plot of the data is: "
plt.plot(X, y, 'b+', markersize=5)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

#%% Compute cost
X = c_[ones((n1, 1)), X]  # tricky!
theta = zeros((2, 1))
# print X

def h_x(X, theta):
    return X.dot(theta)

# print h_x(X, theta)
# error = h_x(X, theta) - y # what the hell?!  h_x is a 2-D array, y is 1-D, so the result is 2-D

def computeCost(X, y, theta):
    m = len(y)
    err = ( h_x(X, theta) - y)
    return 1.0 / (2.0 * m) * err.T.dot(err)

print "\n", "The J(theta) for an initial theta is", computeCost(X, y, theta), "\n"

#%% gradient decent
alpha = 0.01
nIter = 1500 #better results
theta = zeros((2,1))

J_history = zeros((nIter,1))
def gradientDescent(X, y, theta, alpha, nIter):
    m = len(y)
    
    for iter in range(0,nIter):
        err = ( h_x(X,theta) - y )
        theta = theta - alpha*1.0/m * X.T.dot( err )
        J_history[iter,0] = computeCost(X,y,theta)
        
    return theta


theta = gradientDescent(X, y, theta, alpha, nIter)
print "\n", "The estimated theta by gradient descent is ", "\n",
print theta

print "\n", "The convergence of J(theta) is as following:"
plt.plot(range(0,nIter), J_history, '-x')
plt.show()

#%% Normal Equation approach
def normalEqu(X, y):
    return linalg.inv( X.T.dot(X) ).dot(X.T).dot(y)
    
print "\n", "The estimated theta by normal equation is ", "\n",
print normalEqu(X,y)

#%% Plot the fit of the model
print "\n", "The fit of the model is ", "\n"
plt.plot(X[:,1], y, 'rx', markersize=4, label='Data')
plt.plot(X[:,1], X.dot(theta), linewidth = 2, label='fitted')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend(loc='lower right')  # need have labels
plt.show()
