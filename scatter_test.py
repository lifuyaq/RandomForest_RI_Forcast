# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# Import data and print the last few rows
commutes = pd.read_csv('https://raw.githubusercontent.com/blokeley/commutes/master/commutes.csv')

# Create the quantile regression model
model = smf.quantreg('duration ~ prediction', commutes)

# Create a list of quantiles to calculate
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

# Create a list of fits
fits = [model.fit(q=q) for q in quantiles]

# Create a new figure and axes
figure, axes = plt.subplots()

# Plot the scatter of data points
x = commutes['prediction']
axes.scatter(x, commutes['duration'], alpha=0.4)

# Create an array of predictions from the minimum to maximum to create the regression line
_x = np.linspace(x.min(), x.max())

for index, quantile in enumerate(quantiles):
    # Plot the quantile lines
    _y = fits[index].params['prediction'] * _x + fits[index].params['Intercept']
    axes.plot(_x, _y, label=quantile)

# Plot the line of perfect prediction
axes.plot(_x, _x, 'g--', label='Perfect prediction')
axes.legend()
axes.set_xlabel('Predicted duration (minutes)')
axes.set_ylabel('Actual duration (minutes)');

plt.show()
