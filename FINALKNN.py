import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import numpy as np

#STEP 1: CLEAN AND QUANTIFY DATA
sleepData = pd.read_csv('sleep.csv')
sleepData.drop(['Blood Pressure'], axis = 1)
sleepData['Gender'] = sleepData['Gender'].replace({'Male': 0})
sleepData['Gender'] = sleepData['Gender'].replace({'Female': 1})
sleepData['BMI Category'] = sleepData['BMI Category'].replace({'Overweight': 2})
sleepData['BMI Category'] = sleepData['BMI Category'].replace({'Obese': 2})
sleepData['BMI Category'] = sleepData['BMI Category'].replace({'Normal': 1})
sleepData['BMI Category'] = sleepData['BMI Category'].replace({'Normal Weight': 1})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Software Engineer': 0})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Doctor': 1})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Sales Representative': 2})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Teacher': 3})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Nurse': 4})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Engineer': 5})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Accountant': 6})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Scientist': 7})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Lawyer': 8})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Salesperson': 9})
sleepData['Occupation'] = sleepData['Occupation'].replace({'Manager': 10})
sleepData.fillna(0, inplace=True)
sleepData['Sleep Disorder'] = sleepData['Sleep Disorder'].replace({'Sleep Apnea': 1})
sleepData['Sleep Disorder'] = sleepData['Sleep Disorder'].replace({'Insomnia': 2})

#STEP 2: VISUALIZE CLEANED DATA VIA HISTOGRAM
#sleepData.hist(figsize = (8,8))
#plt.show()

#STEP 3: SPLIT DATA AND NORMALIZE
from sklearn.neighbors import KNeighborsClassifier
X = sleepData[['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps']]
Xcopy = sleepData[['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Sleep Disorder']]
y = sleepData[['Sleep Disorder']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#STEP 4: HYPERPARAMETERS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#STEP 4.1: HYPERPARAMETER #1 - COMPARING DISTANCES
distances = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
names = ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski']
eScores = {}
maScores = {}
cScores = {}
miScores = {}
n_neighbors = np.arange(2, 30, 1)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor, metric='euclidean')
    knn.fit(X_test, y_test)
    eScores[neighbor]=knn.score(X_test, y_test)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor, metric='manhattan')
    knn.fit(X_test, y_test)
    maScores[neighbor]=knn.score(X_test, y_test)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor, metric='chebyshev')
    knn.fit(X_test, y_test)
    cScores[neighbor]=knn.score(X_test, y_test)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor, metric='minkowski')
    knn.fit(X_test, y_test)
    miScores[neighbor]=knn.score(X_test, y_test)

eAvg = sum(eScores.values()) / len(eScores)
maAvg = sum(maScores.values()) / len(maScores)
cAvg = sum(cScores.values()) / len(cScores)
miAvg = sum(miScores.values()) / len(miScores)
eLabel = "Euclidean Accuracy. Average = " + str(eAvg)
maLabel = "Manhattan Accuracy. Average = " + str(maAvg)
cLabel = "Chebyshev Accuracy. Average = " + str(cAvg)
miLabel = "Minkowski Accuracy. Average = " + str(miAvg)
plt.plot(n_neighbors, eScores.values(), label= eLabel, linestyle='dashed', linewidth = 2.5, zorder=4)
plt.plot(n_neighbors, maScores.values(), label= maLabel, zorder=3)
plt.plot(n_neighbors, cScores.values(), label= cLabel, zorder=1)
plt.plot(n_neighbors, miScores.values(), label= miLabel, zorder=2)
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracies for Varying Measuring Distances")
plt.legend()
plt.xlim(0, 30)
#plt.ylim(min(allScores)-.02, max(allScores)+.02)
plt.grid()
plt.show()

dictionaries = [eScores, maScores, cScores, miScores]
max_dict = max(dictionaries, key=lambda d: max(d.values()))
best_metric = "minkowski" #DEFAULT
if eScores == max_dict:
    best_metric = "euclidean"
elif maScores == max_dict:
    best_metric = "manhattan"
elif cScores == max_dict:
    best_metric = "chebyshev"
elif miScores == max_dict:
    best_metric = "minkowski"

#STEP 4.3: HYPERPARAMETER #3 - OPTIMAL K NEIGHBOR
train_score = {}
test_score = {}
n_neighbors = np.arange(2, 30, 1)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor, metric=best_metric)
    knn.fit(X_train, y_train)
    train_score[neighbor]=knn.score(X_train, y_train)
    test_score[neighbor]=knn.score(X_test, y_test)

search_key = max(train_score.values())
res = list(train_score.values()).index(search_key)
neighborCount = res + 2

pltTitle = "KNN: Accuracies for Varying Neighbor-Counts using " + best_metric + " Distance"
plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.title(pltTitle)
plt.legend()
plt.xlim(0, 33)
plt.ylim(.6, 1)
plt.grid()
plt.show()

#STEP 5: TRAINING/TESTING MODEL + ACCURACY REPORT
knn = KNeighborsClassifier(n_neighbors=neighborCount, metric=best_metric) #this line uses all previous hyperparameters
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print("FINAL ACCURACY: ", accuracy_scores)
print("FINAL METRIC: ", best_metric)
print("FINAL NEIGHBOR-COUNT: ", neighborCount)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
#TEMPSTEP JUST FOR 10-RUN
knn = KNeighborsClassifier(n_neighbors=neighborCount, metric="euclidean") #this line uses all previous hyperparameters
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print(accuracy_scores)

knn = KNeighborsClassifier(n_neighbors=neighborCount, metric="manhattan") #this line uses all previous hyperparameters
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print(accuracy_scores)

knn = KNeighborsClassifier(n_neighbors=neighborCount, metric="chebyshev") #this line uses all previous hyperparameters
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print(accuracy_scores)

knn = KNeighborsClassifier(n_neighbors=neighborCount, metric="minkowski") #this line uses all previous hyperparameters
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print(accuracy_scores)
'''

#STEP 6: EXTRA VISUALIZATIONS
#STEP 6.1: FEATURE RANKINGS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
dataSleepData = sleepData[['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps']]

X = pd.DataFrame(dataSleepData)
X = X.iloc[:,0:13]

y = sleepData[['Sleep Disorder']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
rf = RandomForestRegressor(n_estimators=neighborCount)
rf.fit(X_train, y_train)
rf.feature_importances_
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
feature_scores = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importanceRank = pd.Series()

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    rf = RandomForestRegressor(n_estimators=neighborCount)
    rf.fit(X_train, y_train)
    rf.feature_importances_
    sorted_idx = rf.feature_importances_.argsort()
    feature_scores = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    importanceRank.loc[feature_scores.index[0]] = rf.feature_importances_
    X = X.drop(feature_scores.index[0], axis=1)
    
print(importanceRank)
print(best_metric)
#STEP 6.2: FEATURE INTERRELATION
corr_matrix = Xcopy.corr()
sns.heatmap(corr_matrix, annot=True)
#plt.show()

