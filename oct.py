"""
OCT Implementation
"""

from gurobipy import *

try:

    # Create a new model
    m = Model("oct")

    # Create variables

    # Set objective

    # Add constraints:

    # m.optimize()

    # for v in m.getVars():
    #     print(v.varName, v.x)
    #
    # print('Obj:', m.objVal)

except GurobiError:
    print('Error reported')