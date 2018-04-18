"""
Example of using a LP file to define model and run the gurobi interactive shell to get the result

The MPS, REW, LP, RLP, ILP, and OPB formats are used to hold optimization models.
"""

import subprocess

result = subprocess.run(['gurobi_cl', 'examples/coins.lp'], stdout=subprocess.PIPE)
output = result.stdout
print(output.decode("utf-8"))
