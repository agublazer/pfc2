import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

dataset = pd.read_csv("MI.data")
dataset = dataset.dropna()
data = dataset.values
# data = data[:, 112:]

"""
for values in data:
	for v in values:
		print(v, end=" ")
	print("\n")
"""


reducer = umap.UMAP()

scaled_data = StandardScaler().fit_transform(data)
embedding = reducer.fit_transform(scaled_data)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    # c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
    c=[0,1,2,3,4,5,6,7,8])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24)
plt.show()