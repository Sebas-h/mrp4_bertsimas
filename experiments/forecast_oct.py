# Load CSV using Pandas
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
import BinTree

# copied from sebas all credits to him
# (Hyper) Parameters
D_max = 1  # maximum depth of the tree
N_min = 6  # minimum number of data points in leaf node
alpha = 0.1  # complexity of the tree


filename = 'data/forecast/forecast.data'
df = pd.read_csv(filename)

# print(df.dtypes)
# print(df)

# convert categorical data to numerical values
char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {}

for c in char_cols:
    df[c], label_mapping[c] = pd.factorize(df[c])

# print(df)
# print(label_mapping)

# convert bool to numerical values
bool_cols = df.dtypes.pipe(lambda x: x[x == 'bool']).index

for c in bool_cols:
    df[c] = df[c].astype(int)

# print(df)



# normalize data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value != min_value:
            # normalize values
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            # all values are identical. set to zero
            result[feature_name] = 0
    return result


def calculate_epsilon(df):
	epsilon = np.array([np.inf] * len(df.columns))

	for feat_no, feat in enumerate(df.columns):
		x = np.sort(df[feat])
		for j in range(len(x) - 1):
			if not x[j+1] == x[j]:
				if x[j+1] - x[j] < epsilon[feat_no]:
					epsilon[feat_no] = x[j+1] - x[j]
	return epsilon


classes = df[['play']].copy()
df = df.drop(['play'], axis = 1)

norm_df = normalize(df)



######### CREATE BINARY TREE ##########
tree = BinTree.generateTree(D_max)

##############

features = [0 for x in range(len(norm_df.columns))]


# for each branch node:


branchNodes = BinTree.get_breadth_first_nodes(tree, True)
for node in branchNodes:
	attribs = []
	for i in range(len(features)):
		attribs.append('a_' + str(i) + '-' + str(node.id))

	# (2)
	# sum of a_t of a branch node is = d_t
	print(' + '.join(attribs) + ' = d_' + str(node.id) )

	# (3)
	# 0 < b_t < d_t 
	print('0 <= ' + 'b_' + str(node.id) + ' <= ' + 'd_' + str(node.id) )


	# (4)
	# a_t are boolean

	# (5)
	# branch node shouldn't split if parent doesn't split
	# d_t =< d_p(t)
	if node.parent is not None:
		print('d_' + str(node.parent.id) + ' =< ' + 'd_' + str(node.id))



leafNodes = BinTree.get_breadth_first_nodes(tree, False, True)
for node in leafNodes:
	sum = []
	# for each leaf node:
	# (6)
	# z_it < l_t
	for i in range(len(norm_df.values)):
		sum.append('z_' + str(i) + '-' + str(node.id))
		print('z_' + str(i) + '-' + str(node.id) + ' =< ' + 'l_' + str(node.id))

	# (7)
	# sum of z_it >= N_min * l_t
	print( ' + '.join(sum) + ' >= ' + 'N_min * l_' + str(node.id))

# for each data point:
# (8)
# sum of z_it (from 1st to last leaf node) = 1
for i in range(len(norm_df.values)):
	datapoints = []
	for node in leafNodes:
		datapoints.append('z_' + str(i) + '-' + str(node.id))
	print(' + '.join(datapoints) + ' = 1')

epsilon = calculate_epsilon(norm_df)
epsilon_max = max(epsilon)
print(epsilon)
print(epsilon_max)

# split constraints:

for node in leafNodes:
	# TODO:
	# get left_splitting_ancesters
	# get right_splitting_ancesters
	ancester = 1
	for j in range(len(features)):
		for i in range(len(norm_df.values)):
			# (9)
			# a_m * (x_i + eps_j =< b_t + (1 + eps_max)(1 - z_it)
			print('a_' + str(ancester) + '-' + str(j) + ' (x_' + str(i) + ' + eps_' + str(j) + ') <= b_' + str(1) + ' + (1 + eps_max)(1 - z_' + str(i) + '-' + str(node.id) + ')'  )


			# (10)
			# a_m * x_i >= b_t - (1- z_it)



# objective function:
# minimize the missclasification loss L_t
# sum of correctly classified - sum of incorrectly classified

# Calculate the 
# Y_ik = 1 (if y_i = k), Y_ik = -1 (otherwise)
# N_kt = 1/2 sum(1 + Y_ik) * z_it
# N_t = sum( z_it )

# label of leaf node
# c_t = max{N_kt}

# constraint
# each leaf node must be corresponding to a class (if it has assigned datapoints)
# sum ( c_kt ) = k_l

# L_t = N_t - max(N_kt) = min( N_t - N_kt )
# linearized
# L_t >= N_t - N_kt - M(1- c_kt)
# L_t <= N_T - N_kt + M_C_kt
# L_t >= 0


