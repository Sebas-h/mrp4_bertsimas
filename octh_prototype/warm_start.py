import math
import pandas as pd
import oct_prototype.preprocessing as preprocessing
from octh_prototype.octh import OCTH


class BranchNode:

    def __init__(self):
        self.id = None
        self.indices_of_data_points = None
        self.a = None
        self.b = None
        self.d = None
        self.a_hat = None
        self.s = None

    def create(self, octh_output, node_id, data_df_indices, num_features):
        self.id = node_id
        self.indices_of_data_points = data_df_indices
        self.a = [octh_output.model.getVarByName('node_{0}_splits_on_feature_{1}'.format(1, j)).X for j in
                  range(num_features)]
        self.b = octh_output.model.getVarByName('split_value_node_{0}'.format(1)).X
        self.d = octh_output.model.getVarByName('node_{0}_applies_split'.format(1)).X
        self.a_hat = [octh_output.model.getVarByName('abs_value_of_node_{0}_splits_on_feature_{1}'.format(1, j)).X for j
                      in range(num_features)]
        self.s = [octh_output.model.getVarByName('feature_{0}_used_in_node_{1}'.format(j, 1)).X for j in
                  range(num_features)]

    def empty_node(self, node_id, num_features):
        self.id = node_id
        self.indices_of_data_points = []
        self.a = [0 for _ in range(int(num_features))]
        self.b = 0
        self.d = 0
        self.a_hat = [0 for _ in range(int(num_features))]
        self.s = [0 for _ in range(int(num_features))]


class LeafNode:

    def __init__(self):
        self.id = None
        self.indices_of_data_points = None
        self.L = None
        self.z = None
        self.n_t = None
        self.n_k_t = None
        self.c_k_t = None
        self.l_t = None

    def create(self, octh_output, node_id, data_df_indices, num_datapoints, class_to_number):
        self.id = node_id
        self.indices_of_data_points = data_df_indices
        self.L = octh_output.model.getVarByName('node_{0}_contains_any_point'.format(node_id)).X
        self.n_t = octh_output.model.getVarByName('total_number_of_points_in_{0}'.format(node_id)).X
        self.l_t = octh_output.model.getVarByName('missclassification_error_in_node_{0}'.format(node_id)).X
        self.z = []
        for i in range(int(num_datapoints)):
            if i in data_df_indices:
                j = data_df_indices.index(i)
                self.z.append(octh_output.model.getVarByName('x{0}_is_in_node_{1}'.format(j, node_id)).X)
            else:
                self.z.append(0.0)
        self.n_k_t = []
        for key, value in class_to_number.items():
            label = None
            for k, v in octh_output.class_to_number.items():
                if key == k:
                    label = v
            if label is not None:
                self.n_k_t.append(octh_output.model.getVarByName(
                    'number_of_points_of_label_{0}_in_node_{1}'.format(label, node_id)).X)
            else:
                self.n_k_t.append(0.0)
        self.c_k_t = []
        for key, value in class_to_number.items():
            label = None
            for k, v in octh_output.class_to_number.items():
                if key == k:
                    label = v
            if label is not None:
                self.c_k_t.append(
                    octh_output.model.getVarByName('label_{0}_is_assigned_to_node_{1}'.format(label, node_id)).X)
            else:
                self.c_k_t.append(0.0)

    def empty_node(self, node_id, num_classes, num_datapoints):
        self.id = node_id
        self.indices_of_data_points = []
        self.L = 0
        self.n_t = 0
        self.l_t = 0
        self.z = [0 for _ in range(int(num_datapoints))]
        self.n_k_t = [0 for _ in range(int(num_classes))]
        self.c_k_t = [0 for _ in range(int(num_classes))]


class BinTree:

    def __init__(self, df, target, num_features, num_datapoints, num_classes, class_to_number, alpha,
                 depth_output_tree):
        self.branch_nodes = []
        self.leaf_nodes = []
        self.df = df
        self.target = target
        self.alpha = alpha
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
        print('### node {0} ###'.format(node_id))
        # Does current node have leaf nodes as child nodes
        has_leafs = node_id > ((2 ** (self.depth_output_tree - 1)) - 1)

        # Select data
        data = self.df.iloc[lambda df1: data_df_indices]

        # Run gurobi; greedy; depth=1
        o = OCTH(data, self.target, tree_complexity=self.alpha, tree_depth=1)
        o.fit(gurobi_print_output=False)

        # Find which indices go to which branch after split
        indices_left_child_node = []
        indices_right_child_node = []
        for i in range(int(o.n_data_points)):
            if o.model.getVarByName('x{0}_is_in_node_{1}'.format(i, 2)).X > 0:
                indices_left_child_node.append(i)
            else:
                indices_right_child_node.append(i)

        # Check if gurobi output makes sense
        if (len(indices_left_child_node) == 0 or len(indices_right_child_node) == 0) and o.model.getVarByName(
                'node_{0}_applies_split'.format(1)).X > 0:
            raise Exception('ERROR: Zero datapoints going down a branch but split variable "d" is not zero.')

        # Current node does not apply a split
        if o.model.getVarByName('node_{0}_applies_split'.format(1)).X < 1:
            leaf_depth = self.depth_output_tree - int(math.floor(math.log2(node_id)))
            branching_depths = leaf_depth - 1

            branch_node_ids = [node_id]
            leaf_node_ids = []

            for d in range(branching_depths):
                for i in range(2 ** (d + 1)):
                    branch_node_ids.append((node_id * (2 ** (d + 1))) + i)

            for i in range(2 ** leaf_depth):
                leaf_node_ids.append((node_id * (2 ** leaf_depth)) + i)

            last_leaf = leaf_node_ids[-1]
            leaf_node_ids = leaf_node_ids[:-1]

            for i in branch_node_ids:
                bn = BranchNode()
                bn.empty_node(i, self.num_features)
                self.add_branch_node(bn)

            for i in leaf_node_ids:
                ln = LeafNode()
                ln.empty_node(i, self.num_classes, self.num_datapoints)
                self.add_leaf_node(ln)

            ln_right_most = LeafNode()
            ln_right_most.create(o, 3, data_df_indices, self.num_datapoints, self.class_to_number)
            ln_right_most.id = last_leaf
            self.add_leaf_node(ln_right_most)

            # Stop recursion
            return

        # Make branch node
        bn = BranchNode()
        bn.create(o, node_id, data_df_indices, self.num_features)
        self.add_branch_node(bn)

        # Current node has leaf nodes as child nodes; make the leaf nodes; stop recursion
        if has_leafs:
            for t in (2, 3):
                ln = LeafNode()
                ln.create(o, t, data_df_indices, self.num_datapoints, self.class_to_number)
                ln.id = (node_id * 2) + (t - 2)
                self.add_leaf_node(ln)
            return

        # Recurse down the tree
        self.rec_greedy_octh(node_id * 2, [bn.indices_of_data_points[index] for index in indices_left_child_node])
        self.rec_greedy_octh((node_id * 2) + 1,
                             [bn.indices_of_data_points[index] for index in indices_right_child_node])
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
