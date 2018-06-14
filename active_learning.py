'''
active_learning.py
translating jerry zhu's active_learning.m into python
'''

import numpy as np
from pdb import set_trace

def compute_invAnegi(A, invA, i):
	'''
	Compute inv(A_neg_i), where A_neg_i is the matrix by removing the ith row/column of A.
	
	Input:
		A: a n*n matrix.
		invA: A^{-1}, a n*n matrix.
		i: the row/column to be removed.
	
	Output:
		invAnegi: the (n-1)*(n-1) matrix inv(A_neg_i).
	
	Algorithm: see section 5 of zhu's 2003 ICML active learning paper
	'''
	n = invA.shape[0]

	# create a perm = (i,1,2,3,...,i-1, i+1, ...)
	perm=np.concatenate((np.array([i]), np.arange(0,i), np.arange(i+1,n)), axis=0) 
	B = A[perm,:][:,perm]
	invB = invA[perm,:][:,perm]

	u=np.concatenate((np.array([-1]), np.zeros((1,n-1)).T), axis=0)
	v=B[0,:] - np.concatenate((np.array([1]), zeros((1,n-1))), axis=0)
	# v = v.T

	tmp = v.dot(invB)
	invBprime = invB + invB[:,0].dot(tmp) / (1 - v.T.dot(invB[:,0]))
	# invBprime = invB - invB * u * v' * invB / (1 + v' * invB * u)

	w = B[:,0]
	w[1] = 0
	tmp = invBprime.dot(w)
	invBprimeprime = invBprime + tmp.dot(invBprime[0,:]) / (1 - invBprime[0,:].dot(w))
	# invBprimeprime = invBprime - invBprime * w * u' * invBprime / (1 + u' * invBprime * w)

	invAnegi = invBprimeprime[1:n,:][:,1:n]

	return invAnegi


def active_learning(nqueries, Y, W, initL):
	'''
	Active learning on top of semi-supervised learning

	Input:
    	nqueries: the desired number of active learning queries, excluding initL
    	Y: n*C matrix, each row in indicator encoding (one 1, others 0). The true labels of all points.
       		Note we don't require labeled points come first. (because we only use Y to get the dimension!)
    	W: n*n graph weights matrix.  
    	initL: index between 1..n.  The initial labeled set. 

  	Output:
    	The following are vectors of length nqueries.
    	query[i]: new queries selected by active learning, given initL
    	acc_ML[i]: The classification accuracy BEFORE adding query i
    	risks[i]: expected risk of query[i], which is the smallest in all possible queries

    Note that python indexing is from 0, instead of 1 which is used in Matlab
	'''

	n, nC = Y.shape
	L = initL
	U = np.setdiff1d(np.arange(0,n), L)

	# the laplacian
	delta = np.diag(np.sum(W, axis=0))

	invDeltaU = np.linalg.inv(delta[U,:][:,U])

	# init outputs
	query = np.zeros(nqueries)
	acc_ML = np.zeros(nqueries)
	risks = np.zeros(nqueries)

	# start active learning, find the best query to ask, add it to the training list, repeat.
	for iteration in range(0,nqueries):
		# compute f the harmonic solution.
		# f is a u*nC matrix
		f = invDeltaU.dot(W(U,L)).dot(Y(L,:))
		# make sure f is a matrix, so not to mess up transpose later on (1d array does not transpose)
		f = np.matrix(f) 

		u = len(U)

		# The classification accuracy BEFORE adding new query
		predicted_class_U = np.where(f==1)[1]
		true_class_U = np.where(Y[U,:]==1)[1]
		acc_ML[iteration] = np.sum(true_class_U==predicted_class_U)/u

		# nG(i,j) = G(i,j)/G(i,i), where G=invDeltaU
		nG = invDeltaU / np.tile(np.diag(invDeltaU).T, (u, 1))

		# we then compute f+(xk, yk), the harmonic function with one more 
		# labeled point (xk,yk).  

		# for efficiency, observe that 
		#     fplus = repmat(f(:,c), 1, u) + repmat(yk_c-f(:,c)', u, 1).*nG
		# can be decomposed into
		#     fplus = repmat(f(:,c), 1, u) - repmat(f(:,c)', u, 1).*nG + yk_c*nG
		# and we can precompute a maxfplus for all c=1:nC without the yk_c*nG term,
		# then for each yk=1:nC, we recompute fplus for c=yk and max it against the precomputed one
		# this is O(nC) instead of O(nC^2).
		for c in range(0,nC):
			# fplus = repmat(f(:,c), 1, u) + repmat(-f(:,c)', u, 1).*nG;
			if c==1:
				# f(:,c) selects a column of f
				# repmat(f(:,c), 1, u) repeats the column u times
				# repmat(-f(:,c)', u, 1) is the transpose of repmat(f(:,c), 1, u)
				pre_maxfplus = np.tile(f[:,c], (1, u)) \
					+ np.tile(-f[:,c].T, (u, 1)).dot(nG)
			else:
				pre_maxfplus = np.maximum(pre_maxfplus, \
					np.tile(f[:,c], (1, u)) + np.tile(-f[:,c].T, (u, 1)).dot(nG))

		risk = np.zeros((1, u)) # the risk of querying point k

		for yk in range(0,nC):
			c = yk
			fplus = np.tile(f[:,c], (1, u)) + np.tile(1-f[:,c].T, (u, 1)).dot(nG)
			maxfplus = np.maximum(fplus, pre_maxfplus)
			risk = risk + np.sum(1-maxfplus).dot(f[:,yk].T)

		# See which example in U, after being queried, will result in the minimum expected risk
		minrisk = np.min(risk)
		minUindex = np.where(risk==minrisk)[0][0]
		query[iteration] = U[minUindex]
		risks[iteration] = minrisk
		L.append(U[minUindex])
		invDeltaU = compute_invAnegi(Delta[U,:][:,U], invDeltaU, minUindex)
		U = np.setdiff1d(U, U[minUindex])

	return query, acc_ML, risks









