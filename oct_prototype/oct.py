import gurobipy
import pandas as pd
import numpy as np
from oct_prototype.classification_tree import BinClassificationTree
import oct_prototype.preprocessing as preprocessing

"""
gurobi workflow:
1. create variables
2. update
3. set objective
4. add constraints
5. optimize
"""

class OCT:
    

    def __init__(self, data, target, tree_complexity, tree_depth, warm_start=False):
        """
        #TODO: Idea: feed in training data and target data separately
        
        data = pandas dataframe holding data
        target = column name of target variable (string) #TODO: also implement integer (col number)
        """
        #create gurobi model
        self.model = gurobipy.Model()
        
        self.data = data
        if isinstance(target, str):
            self.target_var = target
        if isinstance(target, int):
            self.target_var = self.data.columns[target]
        self.not_target_cols = [col for col in self.data.columns if not col==self.target_var]
        self.tree_complexity = tree_complexity # alpha
        self.tree_depth = tree_depth+1 # D
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
        self.c = [] # label assigned to node t
        self.c_k_t = [] # label k is assigned to leaf t
        #objective
        self.cost_matrix = self.create_cost_matrix() # matrix Y
        self.N_k_t = [] # total number of points of label k in node t
        self.N_t = [] #total number of points in node t
        self.L_hat = self._get_baseline_accuracy() #TODO: safety check (=0?)
        self.L_t = [] # missclassification error
        self.tree = None
        
        #self.all_contraints = []        
        self.add_variables()
        
        if warm_start:
            self.warm_start()
        
        self.model.update()
        self.set_objective()
        self.add_constraints()
        
        """print('Variables:')
        print(self.b)
        print(self.d)
        print(self.a)"""
 
    def all_ancestors(self, node):
        """
        returns list of all ancestors of node node
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
        all_ancestors = self.all_ancestors_per_node['node_'+str(node)]
        
        if len(all_ancestors)<1:
            return []
        left_ancestors = []
        
        for ancestor in all_ancestors:
            if ancestor*2==node:
                left_ancestors.append(ancestor)
            node = ancestor
        return left_ancestors
        #return [a for a in self.all_ancestors_per_node['node_'+str(node)] if a%2==0]
    
    def right_ancestors(self, node):
        """
        returns list of right ancestors of node node
        """
        all_ancestors = self.all_ancestors_per_node['node_'+str(node)]
        if len(all_ancestors)<1:
            return []
        right_ancestors = []
            
        for ancestor in all_ancestors:
            if ancestor*2!=node:
                right_ancestors.append(ancestor)
            node = ancestor
        return right_ancestors
        
    def add_variables(self):
        
        #to track a split: a_t and b_t
        for t in self.branch_nodes:
            self.b.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='split_value_node_'+str(t))) #b_t
            self.d.append(self.model.addVar(vtype=gurobipy.GRB.BINARY, lb=0.0, ub=1.0, name='node_{0}_applies_split'.format(t))) #d_t
            #for j in range(self.n_independent_var):
            self.a.append([self.model.addVar(vtype=gurobipy.GRB.BINARY, name='node_{0}_splits_on_feature_{1}'.format(t,j)) for j in range(self.n_independent_var)]) # a_{j,t}

        # to allocate points to leaves    
        for t in self.leaf_nodes:
            self.l.append(self.model.addVar(vtype=gurobipy.GRB.BINARY, name='node_{0}_contains_any_point'.format(t)))
            self.z.append([self.model.addVar(vtype=gurobipy.GRB.BINARY, name='x{0}_is_in_node_{1}'.format(i, t)) for i in range(int(self.n_data_points))])
            
            # for objective function (and track prediction of a node)
            self.N_t.append(self.model.addVar(vtype=gurobipy.GRB.INTEGER, name='total_number_of_points_in_{0}'.format(t)))            
            self.N_k_t.append([self.model.addVar(vtype=gurobipy.GRB.INTEGER, name='number_of_points_of_label_{0}_in_node_{1}'.format(k,t)) for k in range(self.n_classes)])
            #self.c.append(self.model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.n_classes-1, name='label_assigned_to_leaf_{0}'.format(t))) #careful: lb is hardcoded -> classes always need to start from 0
            self.c_k_t.append([self.model.addVar(vtype=gurobipy.GRB.BINARY, name='label_{0}_is_assigned_to_node_{1}'.format(k,t)) for k in range(self.n_classes)])
            
            self.L_t.append(self.model.addVar(vtype=gurobipy.GRB.INTEGER, name='missclassification_error_in_node_{0}'.format(t)))
            
    
    def warm_start(self):
        from cart import CART

        # run cart with same data and congfiguration
        cart = CART(self.data[self.not_target_cols], self.data[self.target_trans])
        cart.run_cart(self.min_number_node, self.tree_depth-1)

        branch_nodes, leaf_nodes, datapoints = cart.translate_tree()

        # set initial values
        for i, bn in enumerate(branch_nodes):
            # set threshold (b_t)
            self.b[i].start = bn.threshold

            # set if branch applies split or not (d_t)
            self.d[i].start = 1 if bn.applies_split else 0

            # set feature to split on (a_{j,t})
            for j in range(self.n_independent_var):
                self.a[i][j].start = 1 if j == bn.split_feature else 0

        for i, ln in enumerate(leaf_nodes):
            # l_t
            self.l[i].start = 1 if ln.num_datapoints > 0 else 0
            # z_it
            for j in range(int(self.n_data_points)):
                self.z[i][j] = 1 if datapoints[j] == ln.id else 0
            
            # N_t
            self.N_t[i].start = ln.num_datapoints
            # N_k_t
            for k in range(self.n_classes):
                self.N_k_t[i][k].start = 0 if ln.value is None else ln.value[0][k]
            # c_k_t
            for k in range(self.n_classes):
                self.c_k_t[i][k].start = 1 if ln.label == k else 0

            # L_t 
            self.L_t[i].start = ln.num_missclass
        
    
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
        #TODO: concat loops for efficiency
        
        #enforce structure of tree:
        # split constraints for a, b, d; formulas (2) and (3)
        for t_no, t in enumerate(self.branch_nodes):
            #t-1 because node numbering starts at 1 and variables lists start at 0
            self.model.addConstr(gurobipy.quicksum(self.a[t_no])==self.d[t_no]) # (2)
            self.model.addConstr(self.b[t_no] >= 0) # (3)
            self.model.addConstr(self.b[t_no] <= self.d[t_no]) # (3)
            #enforce hierarchical structure: (5)
            
            if t!=1:
                parent_node = int(t*0.5)
                self.model.addConstr(self.d[t_no] <= self.d[parent_node-1]) # (5)
        
        #track allocation of points to leaves
        #enumerate because z is only defined for leave nodes
        for t, t_name in enumerate(self.leaf_nodes):
            for i in range(int(self.n_data_points)):
                self.model.addConstr(self.z[t][i] <= self.l[t]) # (6)
            self.model.addConstr(gurobipy.quicksum(self.z[t]) >= self.min_number_node * self.l[t]) # (7)
            #self.model.addConstr(gurobipy.quicksum(self.z[t]) == 1) # (8)
            
            #TODO: efficiency (ouch)
            for i in range(int(self.n_data_points)):
                lin_expr = 0.0
                for t, t_name in enumerate(self.leaf_nodes):
                    lin_expr += self.z[t][i]
                self.model.addConstr(lin_expr == 1) #(8)
                
        
        #constraints enforcing splits: (13) and (14)
        #t_no for accessing z because it is only defined for leaves (indexing)
        
        #epsilon
        epsilon = self.calculate_epsilon()
        epsilon_max=np.max(epsilon)
        epsilon_min = np.min(epsilon)
        
        for t_no, t in enumerate(self.leaf_nodes):
            for i in range(int(self.n_data_points)):
                x_i = self.data[self.not_target_cols].iloc[i].values
                
                #split constraints following formulation in paper
                #for m in self.left_ancestors_per_node.get('node_'+str(t)):
                #    self.model.addConstr(np.dot(self.a[m-1], x_i+epsilon) <= self.b[m-1] + (1+epsilon_max)*(1-self.z[t_no][i])) # (13)
                
                #adding epsilon_min and remove parenthesis
                for m in self.left_ancestors_per_node.get('node_'+str(t)):
                    self.model.addConstr(np.dot(self.a[m-1], x_i)+epsilon_min <= self.b[m-1] + (1+epsilon_max)*(1-self.z[t_no][i])) # (13)
                
                    
                #adding split criterion per feature
                #for feat_no, feat in enumerate(x_i):
                #    for m in self.left_ancestors_per_node.get('node_'+str(t)):
                #        self.model.addConstr(self.a[m-1][feat_no]*x_i[feat_no]+epsilon[feat_no] <= self.b[m-1] + (1+epsilon_max)*(1-self.z[t_no][i]))
                
                for m in self.right_ancestors_per_node.get('node_'+str(t)):
                    self.model.addConstr(np.dot(self.a[m-1], x_i) >= self.b[m-1] - (1-self.z[t_no][i])) #(14)
                    
            # track points and labels assigned to leaf nodes            
            #N_k_t = total number of points of label k in node t
            for k in range(self.n_classes):
                self.model.addConstr(self.N_k_t[t_no][k] == 0.5*sum((1+self.cost_matrix[:,k])*self.z[t_no])) # (15)
                
            #N_t = total number of points in node t
            self.model.addConstr(self.N_t[t_no] == gurobipy.quicksum(self.z[t_no])) # (16)
            
            #c_k_t to track prediction; c_k_t=1 iff c_t=k (label of node t is k); ensure single class prediction for all leafs containing points
            self.model.addConstr(gurobipy.quicksum(self.c_k_t[t_no]) == self.l[t_no]) # (18)
            
            for k in range(self.n_classes):
                #objective: missclassification error
                self.model.addConstr(self.L_t[t_no] >= self.N_t[t_no] - self.N_k_t[t_no][k] - self.n_data_points*(1-self.c_k_t[t_no][k]))
                self.model.addConstr(self.L_t[t_no] <= self.N_t[t_no] - self.N_k_t[t_no][k] + self.n_data_points*self.c_k_t[t_no][k])
                self.model.addConstr(self.L_t[t_no] >= 0) # is this even necessary?
            
            
    def create_cost_matrix(self):
        """
        creates matrix Y from the paper
        """
        Y = np.ones((len(self.data), self.n_classes))*-1
        for row_no, row in enumerate(Y):
            c = int(self.data.iloc[row_no,:][self.target_trans])
            row[c] = 1
        return Y

    def set_objective(self):
        self.model.setObjective((1.0/self.L_hat)*gurobipy.quicksum(self.L_t) + self.tree_complexity*gurobipy.quicksum(self.d))                
        
    def fit(self, time_limit=300, threads=None):
        """
        time_limit: maximum time in seconds for running gurobi optimization
        threads: how many threads to use, if set to None, gurobi default
        """
        self.model.Params.timeLimit = time_limit
        if not threads is None:
            self.model.Params.Threads = threads
        self.model.optimize() #TODO: add stop criterion: if gap hadn't changed for x seconds, then abort
        self.create_tree() # create classification tree
    
    def create_tree(self):
        """
        """
        #model_vars = self.model.getVars()
        self.tree = BinClassificationTree(n_total_nodes=self.tree_max_nodes, model=self.model, n_total_classes=self.n_classes, n_features=self.n_independent_var)
        return
    
    def predict(self, df, feat_cols):
        return self.tree.predict(df, cols=feat_cols)
    
    def training_accuracy(self):
        training_preds = self.predict(self.data, feat_cols=self.not_target_cols)
        actual = self.data[self.target_trans].values
        return sum([1 for i in range(len(training_preds)) if training_preds[i]==actual[i]])/len(actual)
    
    def accuracy_on_test(self, df, target):
        predictions = self.predict(df, feat_cols=self.not_target_cols)
        preds_translated = [self.number_to_class.get(pred) for pred in predictions] #translate back to original labels
        actual = df[self.target_var].values
        return sum([1 for i in range(len(preds_translated)) if preds_translated[i]==actual[i]])/len(actual)
        
    
    def _get_baseline_accuracy(self):
        #return np.max((np.unique(self.data.groupby(by=self.target_var).count().iloc[:,:].values)[-1])) #count of points of most frequent label
        return np.max((np.unique(self.data.groupby(by=self.target_var).count().iloc[:,:].values)[-1]))/self.n_data_points
        #return 1.0/(np.max((np.unique(self.data.groupby(by=self.target_var).count().iloc[:,:].values)[-1]))/self.n_data_points) # don't divide in objective
    

#%%
if __name__=='__main__':
    #target = 'class' #for iris
    target = 4
    df = pd.read_csv('../data/iris/iris.data')
    target_name = df.columns[target]
    print(target_name)
    norm_cols = [col for col in df.columns if not col==target_name]
    
    #filename = '../data/forecast/forecast.data'
    #df = pd.read_csv(filename)
    #target='play'
    print(norm_cols)
    #Preprocessing.categorical_to_numerical(df)
    #Preprocessing.boolean_to_numerical(df)
    preprocessing.normalize(df, norm_cols=norm_cols)
    print(df.head())
    #%%
    
    #%%
    tree_complexity = 0.05
    tree_depth = 1
    df_train, df_test = preprocessing.train_test_split(df, split=0.8)
    print('Training samples: {0}'.format(len(df_train)))
    print('Testing samples: {0}'.format(len(df_test)))
    o = OCT(df_train, target, tree_complexity, tree_depth, warm_start=True)
    #print('Number of independent variables: {0}'.format(o.n_independent_var))
    #print()
    #print('Baseline: {0}\nN_min: {1}'.format(o.norm_constant, o.min_number_node))
    #print('Branch nodes: {0}\nLeaf nodes: {1}'.format(o.branch_nodes, o.leaf_nodes))
    #print()
    #node = 2
    #print('All ancestors of node {0}: {1}'.format(node, o.all_ancestors(node)))
    #print('Left ancestors of node {0}: {1}'.format(node, o.left_ancestors(node)))
    #print('Right ancestors of node {0}: {1}'.format(node, o.right_ancestors(node)))
    #print()
    #print('All ancestors per node: {0}'.format(o.all_ancestors_per_node))
    #print()
    #print('All left ancestors per node: {0}'.format(o.left_ancestors_per_node))
    #print()
    #print('All right ancestors per node: {0}'.format(o.right_ancestors_per_node))
    #print()
    #print('Cost matrix Y:\n{0}'.format(o.cost_matrix))
    #print()
    # o.model.write('oct_example.lp')
    #%%
    o.fit()
    # o.model.write('oct.sol')
    #%%
    print('*'*10)
    print('SOLUTION')
    print('*'*10)
    #v = o.model.getVarByName('label_0_is_assigned_to_node_2')
    #for var in v:
    #%%
    print(o.tree)
    preds = o.predict(df, feat_cols=norm_cols)
    #for node in o.tree.tree.keys():
    #    print(o.tree.tree.get(node).get('node_data'))
    #    print(var.varName, ': ', var.x)
    #%%
    #for pred_no, pred in enumerate(preds):
    #    print('prediction for x{0}: label {1}'.format(pred_no, pred))
    print('Training accuracy: {0}'.format(o.training_accuracy()))
    print('Testing accuracy: {0}'.format(o.accuracy_on_test(df_test,target)))
    #%%
    # o.model.MIPGap