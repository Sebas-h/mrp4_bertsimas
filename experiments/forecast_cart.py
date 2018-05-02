# Load CSV using Pandas
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree


filename = 'data/forecast/forecast.data'
df = pd.read_csv(filename)

# print(df.dtypes)
# print(df)

# convert categorical data to numerical values
char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {}

for c in char_cols:
    df[c], label_mapping[c] = pd.factorize(df[c])

# print(df)
# print(label_mapping)

# convert bool to numerical values
bool_cols = df.dtypes.pipe(lambda x: x[x == 'bool']).index

for c in bool_cols:
    df[c] = df[c].astype(int)

# print(df)



# normalize data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value != min_value:
            # normalize values
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            # all values are identical. set to zero
            result[feature_name] = 0
    return result



classes = df[['play']].copy()
df = df.drop(['play'], axis = 1)

norm_df = normalize(df)

# fit a CART model to the data
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 1)
model.fit(norm_df, classes)

predicted = model.predict(norm_df)

# summarize the fit of the model
print('--------------')
print('model performance')
print('--------------')
print(metrics.classification_report(classes, predicted))
print(metrics.confusion_matrix(classes, predicted))



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
		if (threshold[node] != -2):
			print( indent, "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {" )
			if left[node] != -1:
				recurse (left, right, threshold, features, left[node], depth+1)
				print( indent,"} else {" )
				if right[node] != -1:
					recurse (left, right, threshold, features, right[node], depth+1)
				print( indent,"}" )
		else:
			print( indent,"return " + str(value[node]) )

	recurse(left, right, threshold, features, 0)

print('')
print('--------------')
print('tree structure')
print('--------------')
tree_to_pseudo(model, df.columns.values)