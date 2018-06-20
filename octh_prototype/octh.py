import gurobipy
import pandas as pd
import numpy as np
# from oct_prototype.classification_tree import BinClassificationTree
from binary_tree import BinClassificationTree
import oct_prototype.preprocessing as preprocessing

"""
Gurobi workflow:
0. Create model
1. Create/add variables
2. Update model
3. Set objective
4. Add constraints
5. Optimize
"""


class OCTH:

    def __init__(self, data, target, tree_complexity, tree_depth, warm_start=False):
        """
        data = pandas dataframe holding data
        target = column name of target variable (string)
        """

        # create gurobi model
        self.model = gurobipy.Model()

        self.data = data
        if isinstance(target, str):
            self.target_var = target
        if isinstance(target, int):
            self.target_var = self.data.columns[target]
        self.not_target_cols = [
            col for col in self.data.columns if not col == self.target_var]

        # Hyperparameters
        self.tree_complexity = tree_complexity
        self.tree_depth = tree_depth + 1
        self.n_data_points = len(self.data) * 1.0
        self.min_number_node = int(self.n_data_points * 0.05) * 1.0
        self.norm_constant = self._get_baseline_accuracy()
        self.mu = 0.005

        # Independent variables (i.e. number of features in the dataset)
        self.n_independent_var = len(self.data.columns) - 1

        # Translate class names to numbers
        self.target_trans = '__target__'
        classes = np.unique(self.data[self.target_var])
        self.n_classes = len(classes)
        self.class_to_number = {c: no for no, c in enumerate(classes)}
        self.number_to_class = {no: c for no, c in enumerate(classes)}

        # Add new targets
        self.data[self.target_trans] = self.data[self.target_var].apply(
            lambda c: self.class_to_number.get(c))

        # Max number of nodes
        self.tree_max_nodes = np.power(2, self.tree_depth) - 1  # T
        branch_threshold = int((self.tree_max_nodes + 1) / 2)

        # Sets
        # T_B; index starts from one similar to paper
        self.branch_nodes = [i for i in range(1, branch_threshold)]
        self.leaf_nodes = [i for i in range(
            branch_threshold, self.tree_max_nodes + 1)]  # T_L
        self.all_ancestors_per_node = {
            'node_' + str(i): self.all_ancestors(i) for i in range(1, self.tree_max_nodes + 1)}  # A_R(t)
        self.left_ancestors_per_node = {
            'node_' + str(i): self.left_ancestors(i) for i in range(1, self.tree_max_nodes + 1)}  # A_L(t)
        self.right_ancestors_per_node = {
            'node_' + str(i): self.right_ancestors(i) for i in range(1, self.tree_max_nodes + 1)}  # A(t)

        # Variables to track a split (notation analog to paper)
        self.b = []
        self.d = []
        self.a = []
        self.a_hat = []
        self.s = []

        # Variables to track allocation of points (notation from paper)
        self.l = []  # leaf t contains any point
        self.z = []  # x_i is in node t
        self.c = []  # label assigned to node t
        self.c_k_t = []  # label k is assigned to leaf t

        # Variables needed for objective function
        self.cost_matrix = self.create_cost_matrix()  # matrix Y
        self.N_k_t = []  # total number of points of label k in node t
        self.N_t = []  # total number of points in node t
        self.L_hat = self._get_baseline_accuracy()  # TODO: safety check (=0?)
        self.L_t = []  # missclassification error
        self.tree = None

        # Add variables to gurobi model
        self.add_variables()

        # Add warm start values
        if warm_start:
            self.warm_start()

        # Update model, set objective, and add constraints
        self.model.update()
        self.set_objective()
        self.add_constraints()

    def all_ancestors(self, node):
        """
        returns list of all ancestors of node node
        """
        ancestors = []
        while node > 1:
            node = int(node * 0.5)
            ancestors.append(node)
        return ancestors

    def left_ancestors(self, node):
        """
        returns list of left ancestors of node node
        """
        all_ancestors = self.all_ancestors_per_node['node_' + str(node)]

        if len(all_ancestors) < 1:
            return []
        left_ancestors = []

        for ancestor in all_ancestors:
            if ancestor * 2 == node:
                left_ancestors.append(ancestor)
            node = ancestor
        return left_ancestors
        # return [a for a in self.all_ancestors_per_node['node_'+str(node)] if a%2==0]

    def right_ancestors(self, node):
        """
        returns list of right ancestors of node node
        """
        all_ancestors = self.all_ancestors_per_node['node_' + str(node)]
        if len(all_ancestors) < 1:
            return []
        right_ancestors = []

        for ancestor in all_ancestors:
            if ancestor * 2 != node:
                right_ancestors.append(ancestor)
            node = ancestor
        return right_ancestors

    def add_variables(self):
        # To track a split: a_t and b_t
        for t in self.branch_nodes:
            self.b.append(self.model.addVar(
                vtype=gurobipy.GRB.CONTINUOUS, name='split_value_node_' + str(t)))  # b_t
            self.d.append(self.model.addVar(vtype=gurobipy.GRB.BINARY, lb=0.0,
                                            ub=1.0, name='node_{0}_applies_split'.format(t)))  # d_t
            self.a.append([
                self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=-1.0, ub=1.0,
                                  name='node_{0}_splits_on_feature_{1}'.format(t, j)) for j in
                range(self.n_independent_var)
            ])  # a_{j,t}
            self.a_hat.append([
                self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=1.0,
                                  name='abs_value_of_node_{0}_splits_on_feature_{1}'.format(t, j)) for j in
                range(self.n_independent_var)
            ])  # a_hat{j,t}
            self.s.append([
                self.model.addVar(vtype=gurobipy.GRB.BINARY, lb=0, ub=1.0,
                                  name='feature_{0}_used_in_node_{1}'.format(j, t)) for j in
                range(self.n_independent_var)
            ])  # s{j,t}

        # To allocate points to leaves
        for t in self.leaf_nodes:
            self.l.append(self.model.addVar(vtype=gurobipy.GRB.BINARY,
                                            name='node_{0}_contains_any_point'.format(t)))
            self.z.append([self.model.addVar(vtype=gurobipy.GRB.BINARY, name='x{0}_is_in_node_{1}'.format(
                i, t)) for i in range(int(self.n_data_points))])

            # for objective function (and track prediction of a node)
            self.N_t.append(self.model.addVar(
                vtype=gurobipy.GRB.INTEGER, name='total_number_of_points_in_{0}'.format(t)))
            self.N_k_t.append([self.model.addVar(vtype=gurobipy.GRB.INTEGER,
                                                 name='number_of_points_of_label_{0}_in_node_{1}'.format(k, t)) for k in
                               range(self.n_classes)])
            self.c_k_t.append([self.model.addVar(vtype=gurobipy.GRB.BINARY,
                                                 name='label_{0}_is_assigned_to_node_{1}'.format(k, t)) for k in
                               range(self.n_classes)])

            self.L_t.append(self.model.addVar(
                vtype=gurobipy.GRB.INTEGER, name='missclassification_error_in_node_{0}'.format(t)))

    def add_constraints(self):
        # TODO: concat loops for efficiency
        # enforce structure of tree:
        # split constraints for a, b, d; formulas (2) and (3)
        for t_no, t in enumerate(self.branch_nodes):
            # t-1 because node numbering starts at 1 and variables lists start at 0
            self.model.addConstr(gurobipy.quicksum(self.a_hat[t_no]) <= self.d[t_no])  # (2)
            self.model.addConstr(self.b[t_no] >= -self.d[t_no])  # (3)
            self.model.addConstr(self.b[t_no] <= self.d[t_no])  # (3)
            for j in range(self.n_independent_var):
                self.model.addConstr(self.a[t_no][j] >= -self.s[t_no][j])
                self.model.addConstr(self.a[t_no][j] <= self.s[t_no][j])
                self.model.addConstr(self.s[t_no][j] <= self.d[t_no])
                self.model.addConstr(self.a_hat[t_no][j] >= self.a[t_no][j])
                self.model.addConstr(self.a_hat[t_no][j] >= -self.a[t_no][j])
            self.model.addConstr(gurobipy.quicksum(self.s[t_no]) >= self.d[t_no])
            # enforce hierarchical structure: (5)
            if t != 1:
                parent_node = int(t * 0.5)
                self.model.addConstr(
                    self.d[t_no] <= self.d[parent_node - 1])  # (5)

        # Track allocation of points to leaves
        #   enumerate because z is only defined for leave nodes
        for t, t_name in enumerate(self.leaf_nodes):
            for i in range(int(self.n_data_points)):
                self.model.addConstr(self.z[t][i] <= self.l[t])  # (6)
            self.model.addConstr(gurobipy.quicksum(
                self.z[t]) >= self.min_number_node * self.l[t])  # (7)

            for i in range(int(self.n_data_points)):
                lin_expr = 0.0
                for t, t_name in enumerate(self.leaf_nodes):
                    lin_expr += self.z[t][i]
                self.model.addConstr(lin_expr == 1)  # (8)

        # Constraints enforcing splits: (13) and (14)
        # t_no for accessing z because it is only defined for leaves (indexing)
        for t_no, t in enumerate(self.leaf_nodes):
            for i in range(int(self.n_data_points)):
                x_i = self.data[self.not_target_cols].iloc[i].values
                for m in self.left_ancestors_per_node.get('node_' + str(t)):
                    self.model.addConstr(np.dot(self.a[m - 1], x_i) + self.mu <=
                                         self.b[m - 1] + (2 + self.mu) * (1 - self.z[t_no][i]))  # (13)
                for m in self.right_ancestors_per_node.get('node_' + str(t)):
                    self.model.addConstr(np.dot(self.a[m - 1], x_i) >=
                                         self.b[m - 1] - 2 * (1 - self.z[t_no][i]))  # (14)

            # track points and labels assigned to leaf nodes
            # N_k_t = total number of points of label k in node t
            for k in range(self.n_classes):
                self.model.addConstr(
                    self.N_k_t[t_no][k] == 0.5 * sum((1 + self.cost_matrix[:, k]) * self.z[t_no]))  # (15)

            # N_t = total number of points in node t
            self.model.addConstr(
                self.N_t[t_no] == gurobipy.quicksum(self.z[t_no]))  # (16)

            # c_k_t to track prediction; c_k_t=1 iff c_t=k (label of node t is k);
            #   ensure single class prediction for all leafs containing points
            self.model.addConstr(gurobipy.quicksum(
                self.c_k_t[t_no]) == self.l[t_no])  # (18)

            for k in range(self.n_classes):
                # objective: missclassification error
                self.model.addConstr(
                    self.L_t[t_no] >= self.N_t[t_no] - self.N_k_t[t_no][k] - self.n_data_points * (
                            1 - self.c_k_t[t_no][k]))
                self.model.addConstr(
                    self.L_t[t_no] <= self.N_t[t_no] - self.N_k_t[t_no][k] + self.n_data_points * self.c_k_t[t_no][k])
                # is this even necessary?
                self.model.addConstr(self.L_t[t_no] >= 0)

    def create_cost_matrix(self):
        y = np.ones((len(self.data), self.n_classes)) * -1
        for row_no, row in enumerate(y):
            c = int(self.data.iloc[row_no, :][self.target_trans])
            row[c] = 1
        return y

    def set_objective(self):
        self.model.setObjective(
            (1.0 / self.L_hat) * gurobipy.quicksum(self.L_t) +
            self.tree_complexity * gurobipy.quicksum(
                [gurobipy.quicksum(self.s[t]) for t in range(len(self.branch_nodes))]
            )
        )

    def fit(self, time_limit=300, threads=None):
        """
        time_limit: maximum time in seconds for running gurobi optimization
        threads: how many threads to use, if set to None, gurobi default
        """
        self.model.Params.timeLimit = time_limit
        if not threads is None:
            self.model.Params.Threads = threads
        # TODO: add stop criterion: if gap hadn't changed for x seconds, then abort
        self.model.optimize()
        self.create_tree()  # create classification tree

    def create_tree(self):
        self.tree = BinClassificationTree(n_total_nodes=self.tree_max_nodes, model=self.model,
                                          n_total_classes=self.n_classes, n_features=self.n_independent_var)
        return

    def predict(self, df, feat_cols):
        return self.tree.predict(df, cols=feat_cols)

    def training_accuracy(self):
        training_preds = self.predict(
            self.data, feat_cols=self.not_target_cols)
        actual = self.data[self.target_trans].values
        return sum([1 for i in range(len(training_preds)) if training_preds[i] == actual[i]]) / len(actual)

    def accuracy_on_test(self, df, target):
        predictions = self.predict(df, feat_cols=self.not_target_cols)
        # translate back to original labels
        preds_translated = [self.number_to_class.get(
            pred) for pred in predictions]
        actual = df[self.target_var].values
        return sum([1 for i in range(len(preds_translated)) if preds_translated[i] == actual[i]]) / len(actual)

    def _get_baseline_accuracy(self):
        return np.max(
            (np.unique(self.data.groupby(by=self.target_var).count().iloc[:, :].values)[-1])) / self.n_data_points

    def warm_start(self):
        return 1


if __name__ == '__main__':
    # target = 'class'  # for iris
    target = 4
    df = pd.read_csv('../data/iris/iris.data')
    target_name = df.columns[target]
    print(target_name)
    norm_cols = [col for col in df.columns if not col == target_name]
    print(norm_cols)
    preprocessing.normalize(df, norm_cols=norm_cols)
    print(df.head())
    # Parameters:
    tree_complexity = 2
    tree_depth = 2
    df_train, df_test = preprocessing.train_test_split(df, split=0.8)
    print('Training samples: {0}'.format(len(df_train)))
    print('Testing samples: {0}'.format(len(df_test)))
    o = OCTH(df_train, target, tree_complexity, tree_depth)
    o.model.write('oct_example.lp')
    o.fit()
    o.model.write('oct.sol')
    print('*' * 10)
    print('SOLUTION')
    print('*' * 10)
    print(o.tree)
    preds = o.predict(df, feat_cols=norm_cols)
    print('Training accuracy: {0}'.format(o.training_accuracy()))
    print('Testing accuracy: {0}'.format(o.accuracy_on_test(df_test, target)))
    o.model.MIPGap
