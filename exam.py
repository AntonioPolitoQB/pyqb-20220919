# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: September 19, 2022
#
# You can solve the exercises below by using standard Python 3.10 libraries, NumPy, Matplotlib, Pandas, PyMC3.
# You can browse the documentation: [Python](https://docs.python.org/3.10/), [NumPy](https://numpy.org/doc/stable/user/index.html), [Matplotlib](https://matplotlib.org/stable/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html), [PyMC3](https://docs.pymc.io/en/v3/index.html).
# You can also look at the [slides of the course](https://homes.di.unimi.it/monga/lucidi2122/pyqb00.pdf) or your code on [GitHub](https://github.com).
#
# **It is forbidden to communicate with others.** 
#
#
#
#

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm   # type: ignore
import arviz as az   # type: ignore

# ### Exercise 1 (max 2 points)
#
# The file [butterfly_data.csv](./butterfly_data.csv) (source: https://doi.org/10.13130/RD_UNIMI/5ZXGIV) contains data about a population of butterflies.
#
# Load the data in a pandas dataframe; be sure the columns `organic` and `alternate_management` have the `bool` dtype.

df = pd.read_csv('butterfly_data.csv', sep=',')

df['organic'] = df['organic'].astype(bool)
df['alternate_management'] = df['alternate_management'].astype(bool)

# ### Exercise 2 (max 5 points)
#
# Make a figure with a scatterplot of the `x` and `y` values; each point should be colored according its `subarea`. Use a proper title and a legend (Hint: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html).

#1-astype('category') converts the subarea column to a categorical type.
#2-cat.codes assigns a unique integer code to each category (e.g., 'A' might
#  become 0, 'B' might become 1, etc.).

df['subarea_code'] = df['subarea'].astype('category').cat.codes

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['x'], df['y'], c=df['subarea_code'], cmap='viridis', label=df['subarea'])

# Add a legend and title
plt.legend(*scatter.legend_elements(), title="Subarea")
plt.title('Scatterplot of x and y values colored by subarea')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ### Exercise 3 (max 7 points)
#
# Define a function `distance` that takes two points in a 2D Cartesian plane and returns the Euclidean distance ($d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$) between them. 
#
# To get the full marks, you should declare correctly the type hints and add a test within a doctest string.

import numpy as np

def distance(point1: tuple[float, float], point2: tuple[float, float])->float:
    """
    Calculate the Euclidean distance between two points using numpy.

    >>> distance((0, 0), (3, 4))
    5.0
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2)**2))

# ### Exercise 4 (max 5 points)
#
# Consider the `x` and `y` columns as the coordinates in a plane. Add a column `avg_coll_dist` to the data with the average distance of each butterfly with respect to the other butterflies collected in the same date (year and month).
#
# To get the full marks avoid the use of explicit loops.

#Adding Extra Axes: We add extra axes to align the dimensions of the arrays for broadcasting.
#                  ex: [[1, 2], [3, 4], [5, 6]] has shape: (3,2)
#                      we add an extra axis on position 1, coords[:, np.newaxis, :]  so now it has shape: (3, 1, 2)
#                      [[[1, 2]],
#                       [[3, 4]],
#                       [[5, 6]]]
#                      we add an extra axis on position 0, coords[np.newaxis, :, :]  so now it has shape: (1, 3, 2)
#                      [[[1, 2], [3, 4], [5, 6]]]
#                      Now we have two 3D arrays, one with a 'height' dimension but without a 'width'
#                      dimension (vertical array) and one with a 'width' dimension but without a 'height'
#                      dimension (horizontal array)

#Broadcasting: numpy broadcasts the arrays to a common shape (3, 3, 2), allowing element-wise operations.
#              Considering the two arrays i created, now each operation between them will result in a (3, 3, 2) shape
#              array. The operations are carried out in this order: the first of the vertical array with each element
#              of the horizontal one, then the second and the third element of the vertical array with each element of
#              the horizontal one.

#Result: The result is a 3D array characterized by a grid, where each element represents the difference
#        between the coordinates of two points.

def avg_distance(group):
    coords = group[['x', 'y']].values #I create a df containing only coordinates (group) and then
                                      #I transform it into a 2D array, in which each row contains the
                                      #coordinates of a butterfly
    dist_matrix = np.sqrt(((coords[:, np.newaxis] - coords[np.newaxis, :])**2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.nan) #I don't consider the result when i subtract two equal points
    return np.nanmean(dist_matrix)#I average ignoring Nan values

grouped = df.groupby(['year', 'month'])#I group butterflies by year and month

df['avg_coll_dist'] = grouped.apply(lambda group: avg_distance(group)).reset_index(level=[0,1], drop=True)

#Some values result Nan because there's only one butterfly

#.reset_index(level=[0,1], drop=True) removes the multi-level indexing generated with group.by
#removing (month, year) from the indexing of the result, restoring the original alignment of the dataframe

# ### Exercise 5 (max 3 points)
#
# Print the mean `avg_coll_dist` for each date (month, year).

print(df[['year', 'month', 'avg_coll_dist']])

# ### Exercise 6 (max 3 points)
#
# Plot a histogram with the density of `avg_coll_dist`.

plt.figure(figsize=(10, 6))
plt.hist(df['avg_coll_dist'], bins=30, density=True, alpha=0.6, color='b')
plt.title('Histogram of avg_coll_dist')
plt.xlabel('avg_coll_dist')
plt.ylabel('Density')
plt.show()

# ### Exercise 7 (max 3 points)
#
# Plot together, using two different colors, the histogram with the density of `avg_coll_dist` for `organic` and non-`organic` butterflies.
#

plt.figure(figsize=(10, 6))

# Histogram for organic butterflies
plt.hist(df[df['organic']]['avg_coll_dist'], bins=30, density=True, alpha=0.6, color='b', label='Organic')

# Histogram for non-organic butterflies
plt.hist(df[~df['organic']]['avg_coll_dist'], bins=30, density=True, alpha=0.6, color='r', label='Non-Organic')

# Add title and labels
plt.title('Histogram of avg_coll_dist for Organic and Non-Organic Butterflies')
plt.xlabel('avg_coll_dist')
plt.ylabel('Density')

plt.legend()

plt.show()

# ### Exercise 8 (max 5 points)
#
# Consider this statistical model:
#
#
# - the coordinate `x` divided by 525000 of a butterfly in subarea 'E' is normally distributed with mean $\mu$ and standard deviation $\sigma$ 
# - $\mu$ is normally distributed with mean $=0$ and standard deviation $=5$
# - $\sigma$ is exponentially distributed with $\lambda = 1$
#
# Code this model with pymc3, sample the model, and print the summary of the resulting estimation by using `az.summary`.
#
#
#
#


df_subarea_e = df[df['subarea'] == 'E']

# Normalize the x coordinate by dividing by 525000
x_normalized = df_subarea_e['x'] / 525000

# Define the model
with pm.Model() as model:
    # Priors for unknown model parameters
    mu = pm.Normal('mu', mu=0, sigma=5)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Likelihood (sampling distribution) of observations
    x_obs = pm.Normal('x_obs', mu=mu, sigma=sigma, observed=x_normalized)
    
    # Sample from the posterior
    trace = pm.sample(1000, return_inferencedata=True)

# Print the summary of the resulting estimation
summary = az.summary(trace)
print(summary)

