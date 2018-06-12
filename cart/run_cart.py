# Load CSV using Pandas
import pandas as pd
import numpy as np
import io
from sklearn import metrics
from sklearn import tree
import requests
import re
import time


def train_test_split(df, classes, split=0.8):
    train = df.sample(frac=split)
    train = train.sort_index()
    test = df.drop(train.index)
    train_c = classes.drop(test.index)
    test_c = classes.drop(train.index)

    return train, test, train_c, test_c


def hot_encode(df, target_col, columns):

    hot_encoded_df = pd.get_dummies(df, columns=columns)

    all_cols = list(hot_encoded_df.columns)
    hot_target = all_cols.index(target_col)

    return hot_encoded_df, hot_target


def is_url(string):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        # domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(regex, string) is not None


def run_cart(norm_df, classes, split, D_max, split_criterion, N_min):

    train_df, test_df, train_classes, test_classes = train_test_split(norm_df, classes, split)

    # fit a CART model to the data
    model = tree.DecisionTreeClassifier(
        criterion=split_criterion, max_depth=D_max, min_samples_leaf=N_min)
    model.fit(train_df, train_classes)

    predicted_train = model.predict(train_df)
    predicted_test = model.predict(test_df)

    return model, predicted_train, train_classes, predicted_test, test_classes


# print tree
def tree_to_pseudo(tree, feature_names):
    '''
    Outputs a decision tree model as if/then pseudocode

    Parameters:
    -----------
    tree: decision tree model
            The decision tree to represent as pseudocode
    feature_names: list
            The feature names of the dataset used for building the decision tree
    '''

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth=0):
        indent = "  " * depth
        if threshold[node] != -2:
            print(
                indent, "if ( " + str(features[node]) + " <= " + str(threshold[node]) + " ) {")
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], depth+1)
                print(indent, "} else {")
                if right[node] != -1:
                    recurse(left, right, threshold,
                            features, right[node], depth+1)
                print(indent, "}")
        else:
            print(indent, "return " + str(value[node]))

    recurse(left, right, threshold, features, 0)

# normalize data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value != min_value:
            # normalize values
            result[feature_name] = (
                df[feature_name] - min_value) / (max_value - min_value)
        else:
            # all values are identical. set to zero
            result[feature_name] = 0
    return result


if __name__ == '__main__':
    ##################################
    # SET PARAMETER VALUES
    ##################################

    # (Hyper) Parameters
    D_max = 2  # maximum depth of the tree
    filename = 'data/forecast/forecast.data'
    filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
    header = None
    target_column = 0
    split = 0.75
    repeat = 1000
    best_model = None
    best_accuracy = 0
    verbose = False
    split_criterion = 'gini'  # other value 'entropy'
    hot_encode_cols = None



    ##################################
    # READ AND PREPARE DATA
    ##################################
    character_encoding = 'utf-8'

    if is_url(filename):
        # read dataframe from url
        html = requests.get(filename).content
        s = io.StringIO(html.decode(character_encoding))
        df = pd.read_csv(s, header=header)
    else:
        df = pd.read_csv(filename)

    N_min = int(np.around(df.shape[0] * 0.05, 0))


    # convert categorical data to numerical values
    char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
    label_mapping = {}


    for c in char_cols:
        df[c], label_mapping[c] = pd.factorize(df[c])

    # print(label_mapping)

    # convert bool to numerical values
    bool_cols = df.dtypes.pipe(lambda x: x[x == 'bool']).index

    for c in bool_cols:
        df[c] = df[c].astype(int)


    # print(df)

    #hot encode if needed
    if not hot_encode_cols is None:
        df, target_col = hot_encode(df, target_column, hot_encode_cols)

    classes = df[[target_column]].copy()
    df_data = df.drop([target_column], axis=1)


    norm_df = normalize(df_data)


    startTime = time.time()
    acc_sum = []

    for i in range(repeat):
        m, trp, trc, tp, tc = run_cart(
            norm_df, classes, split, D_max, split_criterion, N_min)
        accuracy = metrics.accuracy_score(tc, tp)
        acc_sum.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = m

        if verbose:
            print('--------------')
            print('model performance')
            print('--------------')
            print('')
            print('Training-Accuracy:')
            print('overall accuracy: ', metrics.accuracy_score(
                trc, trp))
            print(metrics.classification_report(trc, trp))
            print(metrics.confusion_matrix(trc, trp))
            print('')
            print('Test-Accuracy:')
            print('overall accuracy: ', metrics.accuracy_score(
                tc, tp))
            print(metrics.classification_report(tc, tp))
            print(metrics.confusion_matrix(tc, tp))

    avg_acc = np.sum(acc_sum) / repeat

    print('--------------')
    print('Result')
    print('--------------')
    print('Runs: ', repeat)
    print('Runtime: ', time.time() - startTime)
    print('')
    print('Best accuracy: ', best_accuracy)
    print('Average accuracy: ', avg_acc)
    print('')

    # print('')
    # print('--------------')
    # print('tree structure')
    # print('--------------')
    # tree_to_pseudo(best_model, df_data.columns.values)
