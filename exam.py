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
import pymc3 as pm   # type: ignore
import arviz as az   # type: ignore

# ### Exercise 1 (max 2 points)
#
# The file [butterfly_data.csv](./butterfly_data.csv) (source: https://doi.org/10.13130/RD_UNIMI/5ZXGIV) contains data about a population of butterflies.
#
# Load the data in a pandas dataframe; be sure the columns `organic` and `alternate_management` have the `bool` dtype.

pass

# ### Exercise 2 (max 5 points)
#
# Make a figure with two plots: a scatterplot of the `x` and `y` values; each point should be colored according its `subarea`. Use a proper title and a legend (Hint: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html).
#

pass

# ### Exercise 3 (max 7 points)
#
# Define a function `distance` that takes two points in a 2D Cartesian plane and returns the Euclidean distance ($d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$) between them. 
#
# To get the full marks, you should declare correctly the type hints and add a test within a doctest string.

pass

# ### Exercise 4 (max 5 points)
#
# Consider the `x` and `y` columns as the coordinates in a plane. Add a column `avg_coll_dist` to the data with the average distance of each butterfly with respect to the other butterflies collected in the same date (year and month).
#
# To get the full marks avoid the use of explicit loops.

pass   
    

# ### Exercise 5 (max 3 points)
#
# Print the mean `avg_coll_dist` for each date (month, year).

pass

# ### Exercise 6 (max 3 points)
#
# Plot a histogram with the density of `avg_coll_dist`.

pass

# ### Exercise 7 (max 3 points)
#
# Plot together, using two different colors, the histogram with the density of `avg_coll_dist` for `organic` and non-`organic` butterflies.
#

pass

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


pass
