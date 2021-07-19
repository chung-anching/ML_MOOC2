from sklearn import datasets
bunch = datasets.load_breast_cancer()

data = bunch.data
labels = bunch.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, labels,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=labels)

'''----------------------------------------------------------------'''
#Original
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, verbose=1)
'''----------------------------------------------------------------'''
#移除變異數低的特徵
# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(threshold=0.01)
# selector = selector.fit(X_train, Y_train)
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)
# from sklearn.ensemble import RandomForestClassifier
# randomForest = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, verbose=1)
'''----------------------------------------------------------------'''
#單變量特徵選擇
# from sklearn.feature_selection import SelectKBest, chi2
# selector = SelectKBest(chi2,k=20)
# selector = selector.fit(X_train, Y_train)
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)
# from sklearn.ensemble import RandomForestClassifier
# randomForest = RandomForestClassifier(n_estimators=120, criterion='entropy', class_weight='balanced', n_jobs=-1, verbose=1)
'''----------------------------------------------------------------'''

# 進行訓練
randomForest.fit(X_train, Y_train)

# 進行預測
acc = randomForest.score(X_test, Y_test)

print('Accuracy:',acc)
