import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.svm import SVC
from xgboost import plot_tree
from sklearn.metrics import roc_auc_score
import time
start_time = time.time()

data = pd.read_csv("csgo_round_snapshots.csv")
data = data.sample(frac = 1)
data.dropna()

label_encoder = LabelEncoder() 
data["map"]= label_encoder.fit_transform(data["map"])
data["round_winner"]= label_encoder.fit_transform(data["round_winner"])
data["bomb_planted"]= label_encoder.fit_transform(data["bomb_planted"])
for col_name, col_data in data.items():
    if col_data.dtype == "float64":
        data[col_name] = col_data.astype(int)

#data.to_csv("file1.csv")
X = data.drop("round_winner", axis=1)
y = data["round_winner"]

k = 25
chi2_selector = SelectKBest(chi2, k=k) 
X_new = chi2_selector.fit_transform(X, y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
print(metrics.accuracy_score(y_test, clf.predict(X_test)))

from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier


xgb_clf = ('xgb', XGBClassifier(random_state=42, eval_metric=["auc"]))
ada_clf = ('ada', AdaBoostClassifier(clf, random_state=42))

estimators = [
    ('encoder', TargetEncoder()),
    ('scaler', StandardScaler()),
    ('clf', VotingClassifier([xgb_clf, ada_clf], voting="soft")),

]
pipeline = Pipeline(steps=estimators)

pipeline.fit(X_train,y_train)
print(metrics.accuracy_score(y_test, pipeline.predict(X_test)))

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'clf__xgb__n_estimators': Integer(100, 1000),
    'clf__xgb__max_depth': Integer(3, 10),
    'clf__ada__n_estimators': Integer(50, 300),
    'clf__ada__learning_rate': Real(0.1, 1.0, prior='log-uniform'),
}

opt = BayesSearchCV(pipeline, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8) 
print(type(opt))
#xgb = XGBClassifier(scale_pos_weight=SCALE_POS, random_state=1234)
best_params = opt.get_params()
opt.fit(X_train, y_train)
print(opt.best_score_)
print(opt.score(X_test, y_test))

best_estimator = opt.best_estimator_
final = best_estimator.named_steps['clf'].named_estimators_['xgb']

final.fit(X_train, y_train)
print(metrics.accuracy_score(y_test, final.predict(X_test)))
roc=roc_auc_score(y_test,final.predict(X_test))
print("AUC: %.2f%% " % (roc*100))

plot_tree(final)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
