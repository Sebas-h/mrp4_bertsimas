"""
OCT Implementation
"""

from gurobipy import *

try:

    # (Hyper) Parameters
    D_max = 6  # maximum depth of the tree
    N_min = 6  # minimum number of data points in leaf node
    alpha = 0.1  # complexity of the tree

    # Create a new model
    m = Model("oct")

    # Create decision variables
    L_hat = m.addVar(vtype=GRB.BINARY, name="L_hat")
    branch_nodes = []
    leaf_nodes = []
    L = m.addVar(vtype=GRB.INTEGER, name="L")
    d = m.addVar(vtype=GRB.BINARY, name="d")

    # Set objective
    m.setObjective((1 / L_hat), GRB.MINIMIZE)

    # Add constraints:

    # m.optimize()

    # for v in m.getVars():
    #     print(v.varName, v.x)
    #
    # print('Obj:', m.objVal)

except GurobiError:
    print('Error reported')