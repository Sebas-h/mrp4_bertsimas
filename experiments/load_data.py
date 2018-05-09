# Load CSV using Pandas
import pandas
# Import PuLP modeler functions
from pulp import *
from gurobipy import *

filename = 'data/soybeans/soybean-small.data'
df = pandas.read_csv(filename)

classes = df[['class']].copy()
df = df.drop(['class'], axis=1)

print(classes)

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

norm_df = normalize(df)

print(df)
print(norm_df)

# (Hyper) Parameters
D_max = 3  # maximum depth of the tree
N_min = 6  # minimum number of data points in leaf node
alpha = 0.1  # complexity of the tree

# prepare tree
# first level 1 split (2^0), second level 2 splits (2^1), third level 4 splits (2^2) etc.
num_branch_nodes = 0
for x in range(0, D_max):
    num_branch_nodes = num_branch_nodes + 2 ** x

num_leaf_nodes = 2 ** D_max

print("branch-nodes: ", num_branch_nodes)
print("leaf-nodes: ", num_leaf_nodes)


attributes = [0] * ( len(norm_df.columns) - 1)
a = [attributes] * num_branch_nodes


# for each branch node:
# (2)
# sum of a_t of a branch node is = d_t

# (3)
# 0 < b_t < d_t 

# (4)
# a_t are boolean

# (5)
# branch node shouldn't split if parent doesn't split
# d_t =< d_p(t)

# for each branch node:
# (6)
# z_it < l_t

# (7)
# sum of z_it >= N_min * l_t

# for each data point:
# (8)
# sum of z_it (from 1st to last leaf node) = 1

# split constraints:
# (9)
# a_m * x_i < b_t + M_1(1- z_it)
# (10)
# a_m * x_i >= b_t - M_2(1- z_it)
# (11)
# a_m * (x_i + epsilon) < b_t + M_1(1- z_it)


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






# Creates the 'prob' variable to contain the problem data
prob = LpProblem("Soy Beans",LpMinimize)


