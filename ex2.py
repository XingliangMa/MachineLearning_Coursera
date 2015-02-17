# Exercise 2 of Andrew Ng's Machine Learning class on Coursera: 
# Logistic Regression & Regularization
# Only did the first problem. The regulerization is very interesting but it only
# pose minor differences in coding.

# Things learnt: 
# 1. How to use minimization routines in SciPy
# 2. "reshape" is a great tool to reshape arrays. Save a lot of trouble in 
#    dealing with vector vs. matrix
# 3. How to use and specify legend for ploting (size, location)
# 4. It's good to see \n again  (My first programming language is C, and I love it)

# Feb 16, 2015. by Xingliang Ma

from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# set the directory
path = "/Users/horse/Desktop/Coursera/MachineLearning/mlclass-ex2-008/mlclass-ex2/python"
os.chdir(path)

# Problem 1: logistic regression
#%% 1. plot data
data1 = genfromtxt("ex2data1.txt", delimiter=',')
n1 = len(data1)

global X, y   # Not sure if this is a good practice..
X_scores = data1[:, 0:2]  # features
y = data1[:, 2].reshape(n1,1)

negatives = X_scores[ y[:,0]==0 ] # find exam 1 & 2 scores who got admited  
positives = X_scores[ y[:,0]==1 ] # find exam 1 & 2 scores who did NOT get admited

print "\n Plot the data: "
plt.plot( negatives[:, 0], negatives[:, 1] , 'ro', label="Not admitted" )
plt.plot( positives[:, 0], positives[:, 1] , 'b+', label="Admitted" )
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(borderpad=0.01, labelspacing=0.01)
plt.show()

#%% Compute cost
X = c_[ones((n1, 1)), X_scores] 
k = size(X,1)
initial_theta = zeros((k, 1)).reshape(k)

def sigmoid(z):
    return  1 / ( 1.0 + exp(-z) )

def computeCost(theta): # theta has to be the first argument for minimization!!
    theta = theta.reshape(3,1)
    m = len(y)
    h_x = sigmoid( X.dot(theta) )
    return  1.0 / m * ( - log(h_x).T.dot(y) - log(1-h_x).T.dot(1-y) )
    
def grad(theta): 
    theta = theta.reshape(3,1)
    m = len(y)
    h_x = sigmoid( X.dot(theta) )
    return 1.0/m * ( ( h_x -y ).T.dot(X) ).reshape(3)
            
print "\n The J(theta) for an initial theta is \n", computeCost(initial_theta)
print "Its gradient is \n", grad(initial_theta)

#%% minimize the cost function
# In order to use the optimization rountines in Scipy, theta has to be a vector
# Try different initial values. They are actually NOT very robust!!

# no gradient: simplex method
result1 = minimize(computeCost, x0=zeros(k), method='Nelder-Mead') 
print "\n The optimal theta from Nelder-Mead method is: \n", result1.x
print "The final obj fun value is: ", result1.fun

# with gradient BFGS
result2 = minimize(computeCost, x0=zeros(k), method='L-BFGS-B', jac=grad, options={'gtol': 1e-6, 'disp': True}) 
print "\n The optimal theta from L-BFGS-B method is: \n", result2.x
print "The final obj fun value is: ", result2.fun

#%% Plot the decision boundary
theta = result1.x 
print "\n The decision boundary is:"
plt.plot( negatives[:, 0], negatives[:, 1] , 'ro', label="Not admitted" )
plt.plot( positives[:, 0], positives[:, 1] , 'b+', label="Admitted" )
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

plot_x = X[:, 1]
plot_y = - 1/theta[2]  * (theta[0]  + theta[1] * plot_x )  # it is a line
plt.plot( plot_x, plot_y, label = "Decision boundary" )

plt.legend(borderpad=0.01, labelspacing=0.01)
plt.show()

#%% prediction
def predict(theta):
    return sigmoid( X.dot( theta.reshape(3,1) ) ) >= 0.5
    
p = predict(theta)
print "Train Accuracy: ",  (double(p==y)).mean() * 100, "%"
