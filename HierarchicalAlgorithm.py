import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('BankNote_Authentication.csv')
df.dropna(inplace=True)
x = df.iloc[:, :4].values

import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))

plt.title("Dendrogram Plot")
plt.ylabel("Euclidean Distance")
plt.xlabel("Customers")
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred = hc.fit_predict(x)
print(y_pred)