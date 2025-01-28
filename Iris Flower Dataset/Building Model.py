# Loading the required libraries:
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets

# Loading the iris data:
dataset = datasets.load_iris()

# Pre-processing data:

# Creating a variable for the feature data:
X = dataset.data

# Creating a variable for the target data:
y = dataset.target

# Splitting the dataset into training data and validation data:
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Standardising the featured data:

# Loading the standard scaler:
sc = StandardScaler()

# Computing the mean and standard deviation for the training data:
sc.fit(X_train)

# Scaling the training data:
X_train_std = sc.transform(X_train)

# Scaling the test data:
X_test_std = sc.transform(X_validation)

# NOTE: Both of them are scaled to 0 mean and unit variance!

# Standardized feature test data:
print(X_test_std[0:5])

# Building various different kinds of models:

# Split-out validation dataset
array = X_test_std

# Various algorithms:
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]

# Evaluating each model in turn:
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Comparing algorithms graphically:
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# We can clearly see that SVM algorithm was the best one of them all.
# Hence, we will be using it to make predictions!

# Making predictions on validation dataset:
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluating predictions:
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
