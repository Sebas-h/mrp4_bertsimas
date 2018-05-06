# Load CSV using Pandas
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
import BinTree


##################################
# SET PARAMETER VALUES
##################################

# (Hyper) Parameters
D_max = 1  		# maximum depth of the tree
N_min = 1  		# minimum number of data points in leaf node
alpha = 0.1  	# complexity of the tree
filename = 'data/forecast/forecast.data'	# data is read from
output_file = 'experiments/forecast.lp'		# constraints are written to
verbose = False	# toggle console output 


# temporary 
# (to be removed once questions regarding split constraints are clarified)
parenthesis = False
use_epsilon_min = True


##################################
# READ AND PREPARE DATA
##################################

df = pd.read_csv(filename)

# convert categorical data to numerical values
char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {}

for c in char_cols:
    df[c], label_mapping[c] = pd.factorize(df[c])

# convert bool to numerical values
bool_cols = df.dtypes.pipe(lambda x: x[x == 'bool']).index

for c in bool_cols:
    df[c] = df[c].astype(int)

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

# get labels
labels = df[['play']].copy()
df = df.drop(['play'], axis = 1)

# get classes
classes = labels.play.unique()
norm_df = normalize(df)



##################################
# CREATE BINARY TREE 
##################################

tree = BinTree.generateTree(D_max)

features = [0 for x in range(len(norm_df.columns))]
bin_vars = []
generals = []
constraints = []
branchNodes = BinTree.get_breadth_first_nodes(tree, True)
leafNodes = BinTree.get_breadth_first_nodes(tree, False, True)



#####################################
# GENERATE VARIABLES AND CONSTRAINTS 
#####################################

for node in branchNodes:
	attribs = []
	for i in range(len(features)):
		# a_j_m 		j: feature 	m: node
		a = 'a_' + str(i) + '_' + str(node.id)
		attribs.append(a)
		bin_vars.append(a)


	# (2)
	# sum of a_t of a branch node is = d_t
	constraints.append(' + '.join(attribs) + ' - d_' + str(node.id) + ' = 0')
	bin_vars.append('d_' + str(node.id))

	# (3)
	# 0 <= b_t			b_t >= 0 
	# b_t <= d_t		b_t - d_t <= 0
	constraints.append('b_' + str(node.id) + ' >= 0' )
	constraints.append('b_' + str(node.id) + ' - ' + 'd_' + str(node.id) + ' <= 0' )

	# (5)
	# branch node shouldn't split if parent doesn't split
	# d_t <= d_p(t)		d_t - d_p(t) <= 0
	if node.parent is not None:
		constraints.append('d_' + str(node.id) + ' - ' + 'd_' + str(node.parent.id) + ' <= 0' )

	# add b_t generals 
	# (Don't add it to generals. Solver will interpret it as ingeter although it's a continuous value)
	#generals.append('b_' + str(node.id)) 



for node in leafNodes:
	sum = []
	# for each leaf node:
	# (6)
	# z_it < l_t
	for i in range(len(norm_df.values)):
		z = 'z_' + str(i) + '_' + str(node.id)
		sum.append(z)
		bin_vars.append(z)
		constraints.append(z + ' - ' + 'l_' + str(node.id) + ' <= 0' )

	# (7)
	# sum of z_it >= N_min * l_t
	constraints.append( ' + '.join(sum) + ' - ' + str(N_min) + ' l_' + str(node.id) + ' >= 0' )
	bin_vars.append('l_' + str(node.id))


# for each data point:
# (8)
# sum of z_it (from 1st to last leaf node) = 1
for i in range(len(norm_df.values)):
	datapoints = []
	for node in leafNodes:
		datapoints.append('z_' + str(i) + '_' + str(node.id))
	constraints.append(' + '.join(datapoints) + ' = 1')


def calculate_epsilon(df):
	epsilon = np.array([1.0] * len(df.columns))

	for feat_no, feat in enumerate(df.columns):
		x = np.sort(df[feat])
		for j in range(len(x) - 1):
			if not x[j+1] == x[j]:
				if x[j+1] - x[j] < epsilon[feat_no]:
					epsilon[feat_no] = x[j+1] - x[j]
	return epsilon

epsilon = calculate_epsilon(norm_df)
epsilon_max = max(epsilon)
epsilon_min = min(epsilon)


# enforcing split constraints:
# for each leaf node t
for node in leafNodes:
	# get left- and right-splitting ancestors
	ancestors, ancestors_left, ancestors_right = node.get_ancestors()

	# for each left ancestor m (in A_L)
	for ancestor in ancestors_left:

		# for each datapoint i
		for i in range(len(norm_df.values)):
			left_sum = []
			# for each feature j
			for j in range(len(features)):
				if parenthesis:
					left_sum.append( str(norm_df.values[i][j]) + ' a_'  + str(j) + '_' + str(ancestor.id) + ' + ' + str(epsilon[j]) + ' a_'  + str(j) + '_' + str(ancestor.id) )
				elif use_epsilon_min:
					# will create constraints using epsilon_min
					left_sum.append( str(norm_df.values[i][j]) + ' a_'  + str(j) + '_' + str(ancestor.id) )
				else:
					# creates a constraint for every feature using epsilon[j]
					constraints.append( str(norm_df.values[i][j]) + ' a_'  + str(j) + '_' + str(ancestor.id) + ' - b_' + str(ancestor.id) + ' + ' + str(1 + epsilon_max) +  ' z_' + str(i) + '_' + str(node.id)  + ' <= ' + str(1 + epsilon_max - epsilon[j]) )
			
			if len(left_sum) > 0:
				if parenthesis:
					constraints.append( ' + '.join(left_sum) + ' - b_' + str(ancestor.id) + ' + ' + str(1 + epsilon_max) +  ' z_' + str(i) + '_' + str(node.id)  + ' <= ' + str(1 + epsilon_max))
				elif use_epsilon_min:
					constraints.append( ' + '.join(left_sum) + ' - b_' + str(ancestor.id) + ' + ' + str(1 + epsilon_max) +  ' z_' + str(i) + '_' + str(node.id)  + ' <= ' + str(1 + epsilon_max - epsilon_min))


	# for each right ancestor m (in A_R)
	for ancestor in ancestors_right:

		# for each datapoint i
		for i in range(len(norm_df.values)):
			right_sum = []
			# for each feature j
			for j in range(len(features)):
				# (10)
				right_sum.append( str(norm_df.values[i][j]) + ' ' + 'a_'  + str(j) + '_' + str(ancestor.id) )

			if len(right_sum) > 0:  
				constraints.append( ' + '.join(right_sum) + ' - b_' + str(ancestor.id) + ' - z_' + str(i) + '_' + str(node.id) + ' >= -1' )



# objective function:
# minimize the missclasification loss L_t
# sum of correctly classified - sum of incorrectly classified

# prepare helper matrix Y_ik
# Y_ik = 1 (if y_i = k), Y_ik = -1 (otherwise)
y_mat = np.empty([len(labels.values), len(classes)], dtype=int)
for i in range(len(labels.values)):
	# for k in 
	for k in classes:
		if labels.values[i] == k:
			y_mat[i, k] = 1
		else:
			y_mat[i, k] = -1


# N_kt = 1/2 sum(1 + Y_ik) * z_it
for k in classes:
	for t in leafNodes:
		datapoints = []
		for i in range(len(norm_df.values)):
			if y_mat[i, k] > 0:
				datapoints.append(str((1 + y_mat[i, k]) / 2) + ' z_' + str(i) + '_' + str(t.id))
			else:
				datapoints.append(str((1 - abs(y_mat[i, k])) / 2) + ' z_' + str(i) + '_' + str(t.id))
		generals.append('N_' + str(k) + '_' + str(t.id))
		constraints.append('N_' + str(k) + '_' + str(t.id) + ' - ' + ' - '.join(datapoints) + ' = 0' )


# N_t = sum( z_it )
for t in leafNodes:
	datapoints = []
	for i in range(len(norm_df.values)):
		datapoints.append('z_' + str(i) + '_' + str(t.id))
	generals.append('N_' + str(t.id))
	constraints.append('N_' + str(t.id) + ' - ' + ' - '.join(datapoints) + ' = 0')



# each leaf node must be corresponding to a class (if it has assigned datapoints)
# sum ( c_kt ) = l_t
for t in leafNodes:
	cl = []
	for k in classes:
		c = 'c_' + str(k) + '_' + str(t.id)
		cl.append(c)
		bin_vars.append(c)
	constraints.append(' + '.join(cl) + ' - l_' + str(t.id) + ' = 0')


# L_t = N_t - max(N_kt) = min( N_t - N_kt )
M = len(norm_df.values)
for t in leafNodes:
	for k in classes:
		# L_t >= N_t - N_kt - M(1- c_kt)		L_t - N_t + N_kt - M * c_kt >= -M
		constraints.append('L_' + str(t.id) + ' - N_' + str(t.id) + ' + N_' + str(k) + '_' + str(t.id) + ' - ' + str(M) +  ' c_' + str(k) + '_' + str(t.id) + ' >= -' + str(M)  )
		# L_t <= N_t - N_kt + M * C_kt
		constraints.append('L_' + str(t.id) + ' - N_' + str(t.id) + ' + N_' + str(k) + '_' + str(t.id) + ' - ' + str(M) + ' c_' + str(k) +  '_' + str(t.id) + ' <= 0')
	# L_t >= 0
	constraints.append('L_' + str(t.id) + ' >= 0')


# baseline accuracy
majority_class_count = labels['play'].value_counts().max()
L_hat = majority_class_count / len(labels.values)


# objective
# min sum(􏰀L_t) + alfa sum( 􏰀d_t ).
lts = []
for t in leafNodes:
	lts.append('L_' + str(t.id))
	generals.append('L_' + str(t.id))

dts = []
for t in branchNodes:
	dts.append('d_' + str(t.id))



#####################################
# CONSOLE OUTPUT
#####################################

if verbose:
	print('\-------------')
	print('\Data: ')
	print('\-------------')
	print(norm_df)

	print('\-------------')
	print('\Labels: ')
	print('\-------------')
	print(labels)
	print('')

output = []
output.append('\-------------')
output.append('\Objective: ')
output.append('\-------------')
output.append('Minimize' )
lts_m = list(map(lambda x: str(1/L_hat) + ' ' + x , lts))
dts_m = list(map(lambda x: str(alpha) + ' ' + x , dts))
output.append( ' + '.join(lts_m) + ' + ' + ' + '.join(dts_m) )


output.append('')
output.append('\-------------')
output.append('\Constraints: ')
output.append('\-------------')
output.append('Subject To' )
for c in constraints:
	output.append(c)

output.append('')
output.append('\-------------')
output.append('\Variables: ')
output.append('\-------------')
output.append('Binaries')
bin_vars.sort()
output.append(' '.join(bin_vars))
output.append('Generals')
generals.sort()
output.append(' '.join(generals))
output.append('End')

if verbose:
	for o in output:
		print(o)

#####################################
# WRITE LP FILE
#####################################

f = open(output_file, 'w')
for o in output:
	f.write(o)
	f.write('\n')

print('LP file created: ' + output_file)