from sklearn.datasets import load_breast_cancer
bunch = load_breast_cancer()

data = bunch.data
labels = bunch.target
feature_names =bunch.feature_names

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=labels)

from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(verbose=1, n_jobs=-1)

logisticRegression = logisticRegression.fit(X_train, y_train)
accuracy = logisticRegression.score(X_test, y_test)
print("Accuracy:", accuracy)

#----------------------------------------------------------------
#添加class_weight
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
class_weight = dict(zip(unique, counts))

from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(verbose=1, n_jobs=-1,
                                        class_weight=class_weight)

logisticRegression = logisticRegression.fit(X_train, y_train)
accuracy = logisticRegression.score(X_test, y_test)
print("Accuracy:", accuracy)

#----------------------------------------------------------------