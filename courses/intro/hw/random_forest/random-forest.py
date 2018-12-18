import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as slms
from sklearn.metrics import r2_score

data = pd.read_csv("abalone.csv")
data.Sex = data.Sex.replace(['F', 'I', 'M'], [-1, 0, 1])
target = data.Rings
data.drop('Rings', axis='columns', inplace=True)

score = {i: 0 for i in range(1, 51)}
fold = slms.KFold(n_splits=5, shuffle=True, random_state=1)
for ntrees in range(1, 51):
    rfr = RandomForestRegressor(ntrees, random_state=1)
    rfr.fit(data, target)
    score[ntrees] = r2_score(target, slms.cross_val_predict(rfr, data, target, cv=fold))

for key, value in score.items():
    if value > 0.52:
        print("Minimal # of trees such that r2 > 0.52 is ", key)
        break

plt.figure(figsize=(20, 10), dpi=200)
plt.plot(np.arange(1, 51), list(score.values()), color='blue')
plt.hlines(y=0.52, xmin=1, xmax=50, color='red')
plt.show()
