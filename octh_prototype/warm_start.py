import gurobipy
import pandas as pd
import numpy as np
# from oct_prototype.classification_tree import BinClassificationTree
# from octh_prototype.binary_tree import BinClassificationTree
import oct_prototype.preprocessing as preprocessing
from octh_prototype.octh import OCTH


def greedy_octh(data):
    # DATA:
    target = 4
    df = pd.read_csv('../data/iris/iris.data')
    target_name = df.columns[target]
    norm_cols = [col for col in df.columns if not col == target_name]
    preprocessing.normalize(df, norm_cols=norm_cols)

    octh_depth = 2
    points_in_nodes = {1: df}  # starting at node 2 and 3 (next 4, 5, etc)

    for n in range(2 ** octh_depth - 1):
        # n = number of current node
        # this = n + 2**current_depth

        df_train, df_test = preprocessing.train_test_split(df)
        o = OCTH(df_train, target, tree_complexity=0, tree_depth=1)
        o.fit()
        # which data point in which leaf node
        for i in range(int(o.n_data_points)):
            var_name = 'x{0}_is_in_node_{1}'.format(i, 2)
            v = o.model.getVarByName(var_name).X
            if v > 0:
                points_in_nodes[0].append(i)
            else:
                points_in_nodes[0].append(i)

        # data_node_2 = df.loc[points_in_node_2]
        # data_node_3 = df.loc[points_in_node_3]

    # get all variables needed out, for the x-th split in the tree
    #

    # print(o.tree)
    # preds = o.predict(df, feat_cols=norm_cols)
    # print('Training accuracy: {0}'.format(o.training_accuracy()))
    # print('Testing accuracy: {0}'.format(o.accuracy_on_test(df_test, target)))


if __name__ == "__main__":
    greedy_octh(1)
