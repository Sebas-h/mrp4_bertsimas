from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn import tree

# load the iris datasets
dataset = load_iris()
# fit a CART model to the data
model = tree.DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))