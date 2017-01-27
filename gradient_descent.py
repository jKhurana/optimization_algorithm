import numpy as np


#various loss functions



def meanSquareError(x,y,w):
	estimated = np.matmul(np.transpose(x),w)
	error = np.subtract(estimated,y)
	return np.matmul(x,error)
	

# Logistic regression
def logistic(x):
	one = np.array([1])
	e = np.exp(np.negative(x))
	return np.divide(one,np.add(one,e))




def logisticLoss(x,y,w):
	one = np.array([1])
	estimated = logistic(np.matmul(np.transpose(x),w))
	error = np.multiply(np.multiply(estimated-y,estimated),np.subtract(one,estimated))
	return np.matmul(x,error)


# Function for gradient descent algorithm
# (x,y) training data (datapoints x dim,1 x datapoints)
# w0 inital value of parameter
# learning rate (scaler)
# f -: function to calculate gradient of the function which needs to be minimized 
def gradientDescent(x,y,w0,learningrate,eps_err,f):
	wPrev = w0
	print wPrev
	gradient = f(x,y,wPrev)
	wNext = np.subtract(wPrev, np.multiply(learningrate,gradient))
	while np.all(np.fabs(np.subtract(wPrev,wNext)) > eps_err):
		wPrev = wNext
		gradient = f(x,y,wPrev)
		print wNext
		wNext = np.subtract(wPrev,np.multiply(learningrate,gradient))
	return wNext


#testing of the gradient descent algorithm using linear regression
x = np.ones((2,10))
x[1,:] = np.random.rand(10)
y = np.random.randint(2,size=10)

#define terminating criteria
eps_err = np.array([0.01] * 2)

# call gradient descent for meansquare error
learnedWeightsMeanSquareError = gradientDescent(x,y,np.array([1,1]),0.1,eps_err,meanSquareError)
print "Final weights are :"
print learnedWeightsMeanSquareError

#call gradient descent for logistic regression
learnedWeightsLogisticError = gradientDescent(x,y,np.array([1,1]),0.1,eps_err,logisticLoss)
print "Final weights are :"
print learnedWeightsLogisticError