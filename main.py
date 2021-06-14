import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# Implementacion inicial
# Prueba de reduccion de dimensionalidad para visualzacion usando umap y bokeh


def load_csv(url):
	dataset = pd.read_csv(url)
	dataset = dataset.dropna()
	return dataset

penguins = load_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

reducer = umap.UMAP()

# Usar dataset de prueba con 4 atributos
penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values

print(penguin_data)
# Escalar datos
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)


embedding = reducer.fit_transform(scaled_penguin_data)

# Asignar un color a cada elemento del conjunto de datos
# La etiqueta de clase se obtiene de penguins.species_short
colors = [sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})]

# Usar matplotlib
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24)
print(embedding[:, 0])
plt.show()
print(colors)

# Generar visualización interactiva con bokeh

from bokeh.plotting import figure, output_file, show

output_file("square.html")
colors2 = [x for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})]

p = figure(plot_width=400, plot_height=400)

color_maps = {0: "red", 1: "green", 2: "blue"}
for i in range(len(embedding)):
	p.circle(embedding[i, 0], embedding[i, 1], size=5, color=color_maps[colors2[i]], alpha=0.5)

show(p)


############3 Modelos a usar ######################

# Datos entrenamiento
X = penguin_data
Y = colors2

# Decision tree visualization
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

tree.plot_tree(clf)

# Random forest
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X, Y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X, Y)