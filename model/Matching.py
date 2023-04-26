import pulp

import numpy as np


def Bipartite_matching(Weights):
	
	# model

	model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

	# variables

	X = pulp.LpVariable.dicts("X",((O,D) for O in range(Weights.shape[0]) for D in range(Weights.shape[1])),lowBound=0,upBound=1,cat='Integer')

	# objective

	model += (pulp.lpSum([Weights[O,D] * X[(O, D)] for O in range(Weights.shape[0]) for D in range(Weights.shape[1])]))

	# constraints

	for O in range(Weights.shape[0]):

		model += pulp.lpSum([X[(O, D)] for D in range(Weights.shape[1])]) <= 1

	for D in range(Weights.shape[1]):

		model += pulp.lpSum([X[(O, D)] for O in range(Weights.shape[0])]) <= 1

	# solvable

	model.solve()
	
	x = np.zeros([Weights.shape[0],Weights.shape[1]])
	
	for var in X:

		var_value = X[var].varValue

		if var_value !=0:

			x [var[0]][var[1]]=1
			
	return x
