"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from regression import (logreg, utils)
# (you will probably need to import more things here)

def test_prediction():
	# initialize
	log_model=logreg.LogisticRegressor(num_feats=3, learning_rate=0.1, tol=0.01, max_iter=10000, batch_size=10)

	# get a set of weights and the x_vector, assign these manually
	weights=log_model.W # get array of weights, which are randomly generated in the model
	x_vector=np.array([-0.1536077, -0.73873096, 0.69515697, 1]) # manually set x_vector

	# manually compute that make predictions is returning the sigmoid transformation of the linear combination between the x_vector and weights
	z=sum(np.multiply(x_vector, weights))
	predicted_prob=1/(1+np.exp(-z))

	# check that our manually calculated prob for the given vector is equal to our make_prediction output
	assert log_model.make_prediction(x_vector)==predicted_prob

	# repeat with a vector of all zeroes; checks if the sigmoid function is being computed correctly as the expected output is 0.5
	x_vector=np.array([0, 0, 0, 0]) # manually set x_vector - at all zeroes, we get 0.5
	z=sum(np.multiply(x_vector, weights))
	predicted_prob=1/(1+np.exp(-z))

	assert log_model.make_prediction(x_vector)==predicted_prob
	assert log_model.make_prediction(x_vector)==0.5

	# pass

def test_loss_function():

	# compare our loss function to sklearn's loss function

	# create manual dataset; data from https://www.pinecone.io/learn/cross-entropy-loss/
	y_true=np.array([0,1,1,0,0,1,1])
	y_pred=np.array([0.07,0.91,0.74,0.23,0.85,0.17,0.94])

	# initialize
	log_model=logreg.LogisticRegressor(num_feats=3, learning_rate=0.1, tol=0.01, max_iter=10000, batch_size=10)

	# check that our function matches sklearn output
	assert log_model.loss_function(y_true, y_pred)==log_loss(y_true, y_pred)

	# pass

def test_gradient():

	# initialize
	log_model=logreg.LogisticRegressor(num_feats=2, learning_rate=0.1, tol=0.01, max_iter=10000, batch_size=10)

	# get initialized weights from our model and manually generate x_vector
	weights=log_model.W # get array of weights, which are randomly generated in the model
	x_vector=np.array([[-0.1536077, 0.69515697, 1], 
					[-0.534536, 0.034259, 1]]) # manually set x_vector
	y_true=np.array([1, 0]) # define 

	# manually compute weight gradient
	z=np.sum(np.multiply(weights, x_vector), axis=1)
	y_pred=1/(1+np.exp(-z))
	difference=y_pred-y_true
	dot_out=np.dot(difference, x_vector)
	weight_gradient=(1/len(y_true))*(dot_out)

	# check our manual weight gradient equals the models version
	assert np.array_equal(weight_gradient, log_model.calculate_gradient(y_true, x_vector))

	# pass

def test_training():

	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	# initialize model and train
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=10000, batch_size=10)
	log_model.train_model(X_train, y_train, X_val, y_val)

	# get history of weights 
	weight_hist=np.array(log_model.weight_hist)

	# get all adjacent pairs of indices within our weight history, we are aiming to compare each weight vector with the next vector to see if they are different (indicating that they are updating during training)
	pairs=[(i,i+1) for i in range(len(weight_hist)) if (i+1<len(weight_hist))]

	# check if the next array is equal for set of all weights (should be all False)
	updating_weight_hist=[np.array_equal(weight_hist[p[0]], weight_hist[p[1]]) for p in pairs]

	# check that our array contains only false
	assert not any(updating_weight_hist)

	# pass