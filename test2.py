import umap
import umap.plot
from sklearn.datasets import load_digits

# Usando digist y umap plotting 
digits = load_digits()

mapper = umap.UMAP().fit(digits.data)
umap.plot.points(mapper, labels=digits.target)
