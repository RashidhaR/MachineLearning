import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('BankNote_Authentication.csv')
print(df.to_string())

x=df.iloc[:,:4].values
df2=pd.DataFrame(x)

from sklearn.cluster import KMeans
wcss_list=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

print(wcss_list)
plt.plot(range(1,11),wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)

