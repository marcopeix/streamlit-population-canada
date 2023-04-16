A simple web application to visualize the population of Canada and forecast it. Visit it [here](https://marcopeix-streamlit-population-canada-main-4rr347.streamlit.app/)

Data source:  Statistics Canada. [Table 17-10-0009-01  Population estimates, quarterly](https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000901)

# About
This is my first time working with Streamlit and deploying a data application. 

## Objective
The goal of this project is to visualize the quarterly population estimates in Canada and forecast it. 

The app contains an interactive choropleth map, allowing us to select the quarter and the year.

We can also see how the population changes over time, and how the population of different locations compare to each other across time with an animated bar plot

## Forecasting
For the forecasting portion, three models are tested: an autoregressive model, double exponential smoothing, and the Theta model. 

The user sets the target and the horizon (in number of quarters). This launches a function that tests the performance of each model on the specified horizon, using a hold-out test containing 32 timesteps. The models are evaluated using the sMAPE. Then, the model that achieves the lowest sMAPE is used to actually generate the forecast.

The predictions are shown in a plot and we can visualize the performance of each model in a bar plot.