# Loading the required libraries:
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

# Loading the required dataset from the internet:
url = "http://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Numerical Summary of the dataset:

# Shape of the dataset:
print(dataset.shape)

# First 15 rows of the dataset:
print(dataset.head(15))

# Statistical Summary of the dataset:
print(dataset.describe())

# Class distribution of the dataset:
print(dataset.groupby('class').size())

# Graphical Summary of the dataset:

# Box plot:
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# Histogram:
sns.FacetGrid(dataset, hue="class", height=5) \
   .map(sns.distplot, "petal-length") \
   .add_legend()
plt.show()

# Scatter plot matrix:
scatter_matrix(dataset)
pyplot.show()


# Scatter plot:
dataset.plot(kind='scatter', x='petal-length', y='petal-width')
plt.show()

# Bar plot:
a = dataset['class'].value_counts()
species = a.index
count = a.values
plt.bar(species,count,color = 'pink')
plt.xlabel('species')
plt.ylabel('count')
plt.show()

# Heat map:
correlation = dataset.corr()

fig ,ax = plt.subplots()
k = ax.imshow(correlation, cmap = 'magma_r')

ax.set_xticks(np.arange(len(correlation.columns)))
ax.set_yticks(np.arange(len(correlation.columns)))
ax.set_xticklabels(correlation.columns)
ax.set_yticklabels(correlation.columns)

cbar = ax.figure.colorbar(k, ax=ax)
cbar.ax.set_ylabel('color bar', rotation=-90, va="bottom")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

for i in range(len(correlation.columns)):
  for j in range(len(correlation.columns)):
    text = ax.text(j, i, np.around(correlation.iloc[i, j],decimals=2),ha="center", va="center", color="purple")
plt.show()

# Pie-chart:
a= dataset['class'].value_counts()
species = a.index
count = a.values
colors= ['cyan','teal','gold']
explode = (0,0.2,0)
plt.pie(count, labels=species,shadow=True, colors=colors,explode = explode, autopct='%1.1f%%')
plt.xlabel('species')
plt.axis('equal')
plt.show()

# Run-chart:
sns.set_style('darkgrid')
sns.lineplot(data=dataset.drop(['class'], axis=1))
plt.show()

# Pair-plot:
sns.pairplot(dataset, hue="class", height = 5);
plt.show()

# Violin Plot:
sns.violinplot(x="class",y="petal-length",data=dataset)
plt.show()

# 4-D Plot:
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='petal_length', symbol='species')
fig.show()
