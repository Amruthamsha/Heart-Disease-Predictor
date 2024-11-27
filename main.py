import graphviz
import pandas as pd
from IPython.core.display_functions import display
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import export_graphviz
import math


#file opened
df = pd.read_csv('heart-disease.csv')

#pre-processor mapping
df["fbs"] = df["fbs"].map({"yes": 1, "no": 0, "unknown": 0})

#data splitting
X = df.drop("target", axis=1)
y = df["target"]

#training and testing models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#fitting and evaluating model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#visualization of results - first 3 decision trees
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree, feature_names=X_train.columns, filled=True, max_depth=2, impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

#hyperparameter tuning
param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)

#best model
best_rf = rand_search.best_estimator_
#best hyperparameter
print("best hyperparameter: ", rand_search.best_params_)

#confusion matrix

#generate prediction with best model
y_pred = best_rf.predict(X_test)

#create confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

#evaluate accuracy, recall, precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy = {round(accuracy*100, 2)}%")
print(f"Precision = {round(precision*100)}%")
print(f"Recall = {round(recall*100)}%")

#create a series of feature's importance and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns.sort_values(ascending=False))

#plot bar graph
feature_importances.plot.bar()