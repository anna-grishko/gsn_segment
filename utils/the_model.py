import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings

from curve_fitting_utils import sample_equidist
from ef_functions import ef_coefs

sns.set_theme(style="ticks", font_scale=1.25)
random_state = 123
class_dict = {'F': 1, 'M': 0}

# read all data
folder_main = '../data/segmented_curves/final'
sup_folders = os.listdir(folder_main)
# read data into a table
data = []
for folder in sup_folders:
    sub_folders = os.listdir(os.path.join(folder_main, folder))
    for specimen in sub_folders:
        # read registered GSN curves
        semilandmarks_path = os.path.join(folder_main, folder, specimen, 'gsn_curve_registered.csv')
        semilandmarks = pd.read_csv(semilandmarks_path)
        if semilandmarks.values[0, 1] > semilandmarks.values[-1, 1]:
            semilandmarks = semilandmarks.loc[::-1, :].reset_index(drop=True)

        coordinates = sample_equidist(semilandmarks.to_numpy(), 100)
        # close the outline
        new_dat = np.array([np.zeros(coordinates.shape[0]), coordinates[::-1, 1]]).T
        new_dat = new_dat[1:(coordinates.shape[0] - 2), :]  # remove first and last points (duplicities)
        outline = np.concatenate((coordinates, new_dat), axis=0)
        print('filename ' + specimen)
        coefs = ef_coefs(outline, n_harmonics=7)

        df = pd.DataFrame(data=coefs).T
        df['collection'] = folder
        if folder == 'UNAM':
            df['id'] = specimen.split("_")[1] + "_" + specimen.split("_")[2]  # UNAM
        if folder == 'Maxwell':
            df['id'] = ''.join([str(int(''.join(list(specimen)[0:3]))), list(specimen)[3]])  # Maxwell
        if folder == 'Simon':
            df['id'] = specimen.split("_")[0] + "_" + str(int(specimen.split("_")[1]))  # Simon
        if folder == 'Chiang_Mai':
            df['id'] = specimen
        if folder == "Khon_Kaen":
            df['id'] = specimen.split("-")[specimen.split("-").__len__()-1].split("_")[0] + '/' + specimen.split("-")[specimen.split("-").__len__()-1].split("_")[1]
        data.append(df)


data = pd.concat(data)

# read metadata
meta_data = pd.read_csv("../data/collections_all.csv", sep=";")
new_data = data.merge(meta_data, on=['collection', 'id'])

dataset = new_data.iloc[:, np.r_[0:28]]
y = new_data[['is female']]

svm = SVC(random_state=random_state, probability=True)
param_grid = {'kernel': ('linear', 'rbf'),
              "C": np.logspace(-1, 1, num=5),
              'gamma': np.logspace(-1, 1, num=5)}
scaler = StandardScaler()
clf = GridSearchCV(estimator=svm,  param_grid=param_grid, cv=10)
scaler.fit(dataset.values)
joblib.dump(scaler, 'scaler.save')
X_train = scaler.transform(dataset.values)
y_train = np.array(y).ravel()


with ignore_warnings(category=ConvergenceWarning):
    clf.fit(X_train, y_train)
kf = RepeatedKFold(n_splits=5, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=kf)  # 10-fold cross validation
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
score_original = clf.score(X_train, y_train)
print(f"Original score, all data: {score_original:.2f}")

# save the model to disk
filename = 'the_model.save'
joblib.dump(clf, open(filename, 'wb'))

###
# y_pred = clf.predict(X_train)
# y_prob_M = clf.predict_proba(X_train)[:, 0]  # prob Male
# y_prob_F = clf.predict_proba(X_train)[:, 1]  # prob Female
#
# df = pd.concat([pd.Series(y_pred), pd.Series(y_prob_M), pd.Series(y_prob_F)], axis=1)
# df.columns = ['pred_F', 'prob_M', 'prob_F']
# df = pd.concat([new_data.reset_index(), df], axis=1)
# df.to_csv("pred_svm.csv", index=False)
