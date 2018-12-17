import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import ridge

data = pd.read_csv("close_prices.csv")
dj = pd.read_csv("djia_index.csv", index_col="date")

pca = PCA(10)
pca.fit(data.drop("date", axis='columns'))
partial_sum = np.cumsum(pca.explained_variance_ratio_)
dc = pca.transform(data.drop("date", axis='columns'))
data_comp = pd.DataFrame(dc)

print("# of components to explain 90% of variance : ", np.searchsorted(partial_sum, 0.9, 'right') + 1)
print("Pearson correlation between main (first) component and DJI: ", np.corrcoef(data_comp[0], dj["^DJI"])[0, 1])
print("Name of the most valuable company in main component: ", data.columns.values[np.argmax(pca.components_[0]) + 1])

# visualisation:

plt.figure(figsize=(20, 10))
plt.xticks(range(len(data))[::70], data.date[::70])

# original data
sg = ridge.Ridge()
sg.fit(data.drop("date", axis='columns'), dj["^DJI"])
plt.plot(data.index, sg.predict(data.drop("date", axis='columns')), color='red', linewidth=0.6)

# data with pca reduced number of features
sg = ridge.Ridge()
sg.fit(data_comp, dj["^DJI"])
plt.plot(data.index, sg.predict(data_comp), color='navy', linewidth=0.4)

plt.show()

# it can be seen that models are very close to each other, so dimensions reduction works
