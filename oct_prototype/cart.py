import pandas as pd
import numpy as np
from sklearn import tree
from collections import deque
import bin_tree as bt


class CART:

    def __init__(self, norm_df, classes):
        self.model = None
        self.predictions_train = None
        self.predictions_test = None
        self.tree_depth = 1
        self.tree = None
        self.data = norm_df
        self.classes = classes
        self.oct_tree = None


    def run_cart(self, min_num_datapoints, tree_depth, split_criterion='gini'):
        self.tree_depth = tree_depth

        # fit a CART model to the data
        self.model = tree.DecisionTreeClassifier(criterion=split_criterion, max_depth=self.tree_depth, min_samples_leaf=max(int(min_num_datapoints), 1))
        self.model.fit(self.data, self.classes)

        self.predictions_train = self.model.predict(self.data)

        return self.predictions_train


    def test_performance(self, test_data):

        self.predictions_test = self.model.predict(test_data)

        return self.predictions_test


    def get_model(self):
        return self.model


    def translate_tree(self):
        tree = self.model.tree_
        self.oct_tree = bt.generateTree(self.tree_depth)

        left = tree.children_left
        right = tree.children_right
        threshold = tree.threshold
        feature = tree.feature
        value = tree.value

        # build queue.
        # always start at node 0
        node_queue = deque([0])
        self.oct_tree.cart_id = 0
        self.oct_tree.cart_leaf = False

        index = 1
        while node_queue:
            node_id = node_queue.pop()
            curr_oct_node = self.oct_tree.get_node_by_cart_id(node_id)

            if left[node_id] > 0 and right[node_id] > 0:
                # current node is branch
                curr_oct_node.cart_leaf = False
                curr_oct_node.threshold = threshold[node_id]
                curr_oct_node.applies_split = True
                curr_oct_node.split_feature = feature[node_id]
                # add left child to queue
                left_child = curr_oct_node.get_left_child()
                left_child.cart_id = left[node_id]
                node_queue.appendleft(left[node_id])
                # add right child to queue
                left_child = curr_oct_node.get_right_child()
                left_child.cart_id = right[node_id]
                node_queue.appendleft(right[node_id])
            else:
                # current node is leaf
                curr_oct_node.cart_leaf = True
                curr_oct_node.value = value[node_id]

            index = index + 1

        oct_nodes = bt.get_breadth_first_nodes(self.oct_tree)

        # CART tree might be 'shorter' than OCT tree
        # therefore, we check if cart leaf is oct leaf
        # if not, propagate to rightmost succeeding oct leaf
        for i, n in enumerate(oct_nodes):
            if n.cart_leaf and not n.is_leaf:
                cart_id = n.cart_id
                value_tmp = n.value
                n.cart_id = None
                # get rightmost succeeding oct leaf

                right_leaf = n.get_right_child()
                while right_leaf.get_right_child():
                    right_leaf = right_leaf.get_right_child()

                right_leaf.cart_id = cart_id
                right_leaf.value = value_tmp

        branch_nodes = bt.get_breadth_first_nodes(self.oct_tree, exclude_leaf_nodes=True)

        leaf_nodes = bt.get_breadth_first_nodes(self.oct_tree, exclude_branch_nodes=True)

        # determine label in each leaf node
        for i, l in enumerate(leaf_nodes):
            if not l.value is None and l.value.size:
                l.num_datapoints = np.sum(l.value)
                l.label = np.argmax(l.value)
                l.num_missclass = l.num_datapoints - l.value[0][l.label]

        # determine datapoint assignment
        datapoints = np.empty(self.data.shape[0])
        for i, d in enumerate(self.data.values):
            path = self.model.decision_path([d])
            datapoints[i] = self.oct_tree.get_node_by_cart_id(path.indices[-1]).id

        return branch_nodes, leaf_nodes, datapoints
