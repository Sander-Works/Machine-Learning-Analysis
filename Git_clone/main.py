import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Load the dataset.
cancer = datasets.load_breast_cancer()

pd.set_option('display.max_columns', None)

# Displays
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df.head()

# Putting the dataset into two arrays.
x = cancer.data
y = cancer.target

# Splitting the arrays into random train and test subsets.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

# Heatmap to see the relations between data
ax = plt.axes()
sns.heatmap(data=df.corr(), annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
ax.set_title('HEATMAP')

fig = plt.gcf()
fig.set_size_inches(18, 14)

plt.show()

# Scaling the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Finding the optimal value of k.
k_range = range(1, 50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# Plottign KNN into a graph
plt.suptitle('KNN')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.plot(k_range, scores)
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.show()

# Initializing the machine learning models
logReg_model = LogisticRegression(max_iter=10000)
svc_model = svm.SVC()
KNN_model = KNeighborsClassifier(n_neighbors=16)

# Creating an array for the the models
models = [('Logreg', logReg_model), ('SVM', svc_model), ('KNN', KNN_model)]

results = []
names = []
# looping through to evaluate each model and append the results to results[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Looping through the list, to train each model
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    #  Plotting Confusion Matrix for each model
    cnf_matrix = confusion_matrix(y_test, predictions)
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Pastel2", fmt='g')
    cnf_graph = plt
    cnf_graph.title(name)
    cnf_graph.ylabel('Actual label')
    cnf_graph.xlabel('Prediction label')
    cnf_graph.show()

# Defining the scoring matrixes before the table
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}
kfold = model_selection.KFold(n_splits=10)

# Evaluating the models
log = cross_validate(logReg_model, X_train, y_train, cv=kfold, scoring=scoring)
svm = cross_validate(svc_model, X_train, y_train, cv=kfold, scoring=scoring)
svc = cross_validate(svc_model, X_train, y_train, cv=kfold, scoring=scoring)
knn = cross_validate(KNN_model, X_train, y_train, cv=kfold, scoring=scoring)

# Adding score table to show crossvalidations of models
models_scores_table = pd.DataFrame({'Logistic Regression': [log['test_accuracy'].mean(),
                                                            log['test_precision'].mean(),
                                                            log['test_recall'].mean(),
                                                            log['test_f1_score'].mean()],

                                    'SVM': [svm['test_accuracy'].mean(),
                                            svm['test_precision'].mean(),
                                            svm['test_recall'].mean(),
                                            svm['test_f1_score'].mean()],

                                    'KNN': [knn['test_accuracy'].mean(),
                                            knn['test_precision'].mean(),
                                            knn['test_recall'].mean(),
                                            knn['test_f1_score'].mean()]},

                                   index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Add 'Best Score' column
models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
print(models_scores_table)

# Visualization of scores.
fig = plt.figure()
fig.suptitle("Machine Learing Model Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.show()
