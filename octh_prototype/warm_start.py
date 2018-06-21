import gurobipy
import pandas as pd
import numpy as np
# from oct_prototype.classification_tree import BinClassificationTree
# from octh_prototype.binary_tree import BinClassificationTree
import oct_prototype.preprocessing as preprocessing
from octh_prototype.octh import OCTH


class BranchNode:

    def __init__(self, number, indices_of_data_points, a, b, d, a_hat, s):
        self.number = number
        self.indices_of_data_points = indices_of_data_points
        self.a = a
        self.b = b
        self.d = d
        self.a_hat = a_hat
        self.s = s


class LeafNode:

    def __init__(self, number, indices_of_data_points, l, z, n_t, n_k_t, c_k_t, l_t):
        self.number = number
        self.indices_of_data_points = indices_of_data_points
        self.l = l
        self.z = z
        self.n_t = n_t
        self.n_k_t = n_k_t
        self.c_k_t = c_k_t
        self.l_t = l_t


class BinTree:

    def __init__(self, df, target, num_features, num_datapoints, num_classes, class_to_number, alpha, depth_output_tree):
        self.branch_nodes = []
        self.leaf_nodes = []
        self.df = df
        self.target = target
        self.alpha = alpha
        self.depth_of_greedy_tree = 1
        self.depth_output_tree = depth_output_tree
        self.num_features = num_features
        self.num_datapoints = num_datapoints
        self.num_classes = num_classes
        self.class_to_number = class_to_number

    def add_branch_node(self, node):
        self.branch_nodes.append(node)

    def add_leaf_node(self, node):
        self.leaf_nodes.append(node)

    def rec_greedy_octh(self, node_id, data_df_indices):
        has_leafs = node_id > ((2 ** (self.depth_output_tree - 1)) - 1)
        no_data_points = len(data_df_indices) == 0

        if no_data_points:
            if has_leafs:
                ln = LeafNode(node_id, [], [], [], [], [], [], [])
                self.add_leaf_node(ln)
                self.add_leaf_node(ln)
            indices_left_node = []
            indices_right_node = []
            bn = BranchNode(node_id, [], [], 0, 0, [], [])
            self.add_branch_node(bn)
            if has_leafs:
                return
            self.rec_greedy_octh(node_id * 2, [bn.indices_of_data_points[index] for index in indices_left_node])
            self.rec_greedy_octh((node_id * 2) + 1, [bn.indices_of_data_points[index] for index in indices_right_node])
            return

        data = self.df.iloc[lambda df1: data_df_indices]
        o = OCTH(data, self.target, tree_complexity=self.alpha, tree_depth=self.depth_of_greedy_tree)
        o.fit(gurobi_print_output=False)

        # Stop criteria for recursion: Are we at a node that has leaf nodes as children?
        if has_leafs:
            for t in (2, 3):
                l = o.model.getVarByName('node_{0}_contains_any_point'.format(t)).X
                nt = o.model.getVarByName('total_number_of_points_in_{0}'.format(t)).X
                lt = o.model.getVarByName('missclassification_error_in_node_{0}'.format(t)).X

                z = []
                for i in range(int(self.num_datapoints)):
                    if i in data_df_indices:
                        j = data_df_indices.index(i)
                        z.append(o.model.getVarByName('x{0}_is_in_node_{1}'.format(j, t)).X)
                    else:
                        z.append(0.0)

                nkt = []
                for key, value in self.class_to_number.items():
                    label = None
                    for k, v in o.class_to_number.items():
                        if key == k:
                            label = v
                    if label is not None:
                        nkt.append(o.model.getVarByName('number_of_points_of_label_{0}_in_node_{1}'.format(label, t)).X)
                    else:
                        nkt.append(0.0)

                ckt = []
                for key, value in self.class_to_number.items():
                    label = None
                    for k, v in o.class_to_number.items():
                        if key == k:
                            label = v
                    if label is not None:
                        ckt.append(o.model.getVarByName('label_{0}_is_assigned_to_node_{1}'.format(label, t)).X)
                    else:
                        ckt.append(0.0)

                ln = LeafNode(node_id, data_df_indices, l, z, nt, nkt, ckt, lt)
                self.add_leaf_node(ln)

        # FIND DATA POINTS TO LEFT AND RIGHT NODES AFTER SPLIT
        indices_left_node = []
        indices_right_node = []
        for i in range(int(o.n_data_points)):
            if o.model.getVarByName('x{0}_is_in_node_{1}'.format(i, 2)).X > 0:
                indices_left_node.append(i)
            else:
                indices_right_node.append(i)

        # GET VARIABLE VALUES
        a = [o.model.getVarByName('node_{0}_splits_on_feature_{1}'.format(1, j)).X for j in
             range(self.num_features)]
        b = o.model.getVarByName('split_value_node_{0}'.format(1)).X
        d = o.model.getVarByName('node_{0}_applies_split'.format(1)).X
        a_hat = [o.model.getVarByName('abs_value_of_node_{0}_splits_on_feature_{1}'.format(1, j)).X for j in
                 range(self.num_features)]
        s = [o.model.getVarByName('feature_{0}_used_in_node_{1}'.format(j, 1)).X for j in
             range(self.num_features)]
        bn = BranchNode(node_id, data_df_indices, a, b, d, a_hat, s)
        self.add_branch_node(bn)

        if has_leafs:
            return

        # RECURSION
        self.rec_greedy_octh(node_id * 2, [bn.indices_of_data_points[index] for index in indices_left_node])
        self.rec_greedy_octh((node_id * 2) + 1, [bn.indices_of_data_points[index] for index in indices_right_node])
        return


if __name__ == "__main__":
    # DATA
    target = 4
    df = pd.read_csv('../data/iris/iris.data')
    target_name = df.columns[target]
    norm_cols = [col for col in df.columns if not col == target_name]
    preprocessing.normalize(df, norm_cols=norm_cols)

    # greedy octh
    bt = BinTree(df, target, 0, 3)
    bt.rec_greedy_octh(1, [x for x in range(df.shape[0])])
    print('done with greedy octh warm start')
