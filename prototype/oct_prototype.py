import gurobipy
import pandas as pd
import numpy as np

"""
gurobi workflow:
1. create variables
2. update
3. set objective
4. add constraints
5. optimize
"""

class OCT:
    
    def __init__(self, data, target, tree_complexity, tree_depth):
        """
        data = pandas dataframe holding data
        target = column name of target variable (string) #TODO: also implement integer (col number)
        """
        #create gurobi model
        self.model = gurobipy.Model()
        
        self.data = data
        self.target_var = target
        self.not_target_cols = [col for col in self.data.columns if not col==target]
        self.tree_complexity = tree_complexity # alpha
        self.tree_depth = tree_depth # D
        self.n_data_points = len(self.data)*1.0 #total number of training points
        self.norm_constant = self._get_baseline_accuracy() #L hat
        self.min_number_node = int(self.n_data_points*0.05)*1.0 #N_min
        
        # independent variables
        self.n_independent_var = len(self.data.columns)-1
        #translate class names to numbers
        self.target_trans = '__target__'
        classes = np.unique(self.data[self.target_var])
        self.n_classes = len(classes)
        self.class_to_number = {c:no for no, c in enumerate(classes)}
        self.number_to_class = {no:c for no, c in enumerate(classes)}
        #add new targets
        self.data[self.target_trans] = self.data[self.target_var].apply(lambda c: self.class_to_number.get(c))
        
        #max number of nodes
        self.tree_max_nodes = np.power(2, self.tree_depth)-1 # T
        branch_threshold = int((self.tree_max_nodes+1)/2)
        
        #sets
        self.branch_nodes = [i for i in range(1, branch_threshold) ] #T_B; index starts from one similar to paper
        self.leaf_nodes = [i for i in range(branch_threshold, self.tree_max_nodes+1)] #T_L        
        self.all_ancestors_per_node = {'node_'+str(i):self.all_ancestors(i) for i in range(1, self.tree_max_nodes+1)} # A_R(t)
        self.left_ancestors_per_node = {'node_'+str(i):self.left_ancestors(i) for i in range(1, self.tree_max_nodes+1)} # A_L(t)
        self.right_ancestors_per_node = {'node_'+str(i):self.right_ancestors(i) for i in range(1, self.tree_max_nodes+1)} # A(t)
        
        #to track a split (notation analog to paper)
        self.b = []
        self.d = []
        self.a = []
        #to track allocation of points (notation from paper)
        self.l = [] # leaf t contains any point
        self.z = [] # x_i is in node t
        
        self.all_contraints = []        
        self.add_variables()
        self.model.update()
        self.set_objective()
        self.add_constraints()
        self.model.update()
        
        #objective
        self.cost_matrix = self.create_cost_matrix() # matrix Y
        
        """print('Variables:')
        print(self.b)
        print(self.d)
        print(self.a)"""
 
    def all_ancestors(self, node):
        """
        returns list off all ancestors of node node
        """
        ancestors = []
        while node>1:
            node = int(node*0.5)
            ancestors.append(node)
        return ancestors
        
    def left_ancestors(self, node):
        """
        returns list of left ancestors of node node
        """
        return [a for a in self.all_ancestors_per_node['node_'+str(node)] if a%2!=0]
    
    def right_ancestors(self, node):
        """
        returns list of right ancestors of node node
        """
        
        return [a for a in self.all_ancestors_per_node['node_'+str(node)] if a%2==0]
    
    def add_variables(self):
        
        #to track a split: a_t and b_t
        for t in self.branch_nodes:
            self.b.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='split_value_node_'+str(t))) #b_t
            self.d.append(self.model.addVar(vtype=gurobipy.GRB.BINARY, name='node_{0}_applies_split'.format(t))) #d_t
            #for j in range(self.n_independent_var):
            self.a.append([self.model.addVar(vtype=gurobipy.GRB.BINARY, name='node_{0}_considers_var_{1}'.format(t,j)) for j in range(self.n_independent_var)]) # a_{j,t}
        # to allocate points to leaves    
        for t in self.leaf_nodes:
            self.l.append(self.model.addVar(vtype=gurobipy.GRB.BINARY, name='node_{0}_contains_any_point'.format(t)))
            self.z.append([self.model.addVar(vtype=gurobipy.GRB.BINARY, name='x{0}_is_in_node_{1}'.format(i, t)) for i in range(int(self.n_data_points))])
    
    def calculate_epsilon(self):
        epsilon = np.array([np.inf]*len(self.not_target_cols))
        
        for feat_no, feat in enumerate(self.not_target_cols):
            x = np.sort(self.data[feat])
            for j in range(len(x)-1):
                if not x[j+1]==x[j]:
                    if x[j+1]-x[j]<epsilon[feat_no]:
                        epsilon[feat_no]=x[j+1]-x[j]
        return epsilon
    
    def add_constraints(self):
        #enforce structure of tree:
        # split constraints for a, b, d; formulas (2) and (3)
        for t in self.branch_nodes:
            #t-1 because node numbering starts at 1 and variables lists start at 0
            self.model.addConstr(gurobipy.quicksum(self.a[t-1])==self.d[t-1]) # (2)
            self.model.addConstr(0 <= self.b[t-1] <= self.d[t-1]) # (3)
            #enforce hierarchical structure: (5)
            if t!=1:
                parent_node = int(t*0.5)
                self.model.addConstr(self.d[t-1] <= self.d[parent_node-1]) # (5)
    
        #track allocation of points to leaves
        #enumerate because z is only defined for leave nodes
        for t, t_name in enumerate(self.leaf_nodes):
            for i in range(int(self.n_data_points)):
                self.model.addConstr(self.z[t][i] <= self.l[t]) # (6)
            self.model.addConstr(gurobipy.quicksum(self.z[t]) >= self.min_number_node * self.l[t]) # (7)
            self.model.addConstr(gurobipy.quicksum(self.z[t]) == 1) # (8)
        
        #constraints enforcing splits: (13) and (14)
        #t_no for accessing z because it is only defined for leaves (indexing)
        for t_no, t in enumerate(self.leaf_nodes):
            for i in range(int(self.n_data_points)):
                x_i = self.data[self.not_target_cols].iloc[i].values
                epsilon = self.calculate_epsilon()
                epsilon_max=np.max(epsilon)
                for m in self.left_ancestors_per_node.get('node_'+str(t)):
                    self.model.addConstr(np.dot(self.a[m-1], x_i+epsilon) <= self.b[m-1] + (1+epsilon_max)*(1-self.z[t_no][i])) # (13)
                for m in self.right_ancestors_per_node.get('node_'+str(t)):
                    self.model.addConstr(np.dot(self.a[m-1], x_i) >= self.b[m-1] - (1-self.z[t_no][i]))
            

    def create_cost_matrix(self):
        """
        creates matrix Y from the paper
        """
        Y = np.ones((len(self.data), self.n_classes))*-1
        for row_no, row in enumerate(Y):
            c = self.data.iloc[row_no,:][self.target_trans]
            row[c] = 1
        return Y

    def set_objective(self):
        
        return
    
    def create_objective(self):
        return
    
    def create_contraints(self):
        return
    
    def fit(self):
        return
        
    def _get_baseline_accuracy(self):
        return (np.sort(self.data.groupby(by=target).count().iloc[0,:].values)[-1])/self.n_data_points
    

#%%
iris_df = pd.read_csv('iris.data')
target = 'class'
tree_complexity = 0.05
tree_depth = 4
o = OCT(iris_df, target, tree_complexity, tree_depth)
print('Number of independent variables: {0}'.format(o.n_independent_var))
print()
print('Baseline: {0}\nN_min: {1}'.format(o.norm_constant, o.min_number_node))
print('Branch nodes: {0}\nLeaf nodes: {1}'.format(o.branch_nodes, o.leaf_nodes))
print()
node = 5
print('All ancestors of node {0}: {1}'.format(node, o.all_ancestors(node)))
print('Left ancestors of node {0}: {1}'.format(node, o.left_ancestors(node)))
print('Right ancestors of node {0}: {1}'.format(node, o.right_ancestors(node)))
print()
print('All ancestors per node: {0}'.format(o.all_ancestors_per_node))
print()
print('All left ancestors per node: {0}'.format(o.left_ancestors_per_node))
print()
print('All right ancestors per node: {0}'.format(o.right_ancestors_per_node))
print()
#print('Cost matrix Y:\n{0}'.format(o.cost_matrix))
#print()
o.model.write('oct_example.lp')
