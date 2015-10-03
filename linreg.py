import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in xrange(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            #print "Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta
            # TODO:  add update equation here
            h  = np.dot(X,theta)
            diff = h - y
            desc = np.dot(X.T,diff) / X.shape[0]
            theta = theta - (self.alpha * desc)
        self.theta = theta
        return self.theta
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        # TODO: add objective (cost) equation here
        h = np.dot(X,theta)
        d_by_n = (h - y).T
        n_by_d = (h - y)
        dot = np.dot(d_by_n,n_by_d)
        cost = dot / (2 * X.shape[0])
        cost = cost.tolist()[0][0]
        return cost

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        if self.theta==None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # TODO:  add prediction function here
        theta_vector = self.theta
        pred_y = np.dot(X, theta_vector)
        return pred_y