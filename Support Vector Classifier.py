from sklearn import datasets
bunch = datasets.load_breast_cancer()

data = bunch.data
labels = bunch.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

from sklearn.svm import SVC
'''---------------------------------'''
#Original
svc = SVC()
'''---------------------------------'''
#調整模型參數
# svc = SVC(C=1e5, kernel='linear', gamma='scale', class_weight='balanced')
'''---------------------------------'''
# 用RFE,遞迴特徵選擇
# svc = SVC(C=1e5, kernel='linear', gamma='scale', class_weight='balanced')
# from sklearn.feature_selection import RFE
# selector = RFE(estimator=svc, n_features_to_select=27)
# selector = selector.fit(X_train, Y_train)
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)

# 進行訓練
svc.fit(X_train, Y_train)

# 進行預測
acc = svc.score(X_test, Y_test)

print('Accuracy:',acc)

import pickle

# 儲存Model
with open('svc.pickle', 'wb') as f:
    pickle.dump(svc, f)

# 讀取Model
with open('svc.pickle', 'rb') as f:
    svc = pickle.load(f)
    
# 測試讀入的Model
prediction = svc.predict(X_test[0:1])
print(prediction)