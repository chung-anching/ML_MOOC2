from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# 載入資料
bunch = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(bunch.data,bunch.target,
                                                    test_size= 0.3,
                                                    shuffle=True,
                                                    stratify=bunch.target)

# 建立模型
# logreg = LogisticRegression(C=1e5)
logreg = LogisticRegression(C=1e5, class_weight='balanced')

# 用RFE,遞迴特徵選擇
selector = RFE(estimator=logreg, n_features_to_select=27)
selector = selector.fit(X_train, Y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# 進行訓練
logreg.fit(X_train, Y_train)

# 進行預測
acc = logreg.score(X_test, Y_test)
print('Accuracy:', acc)