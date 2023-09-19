# StockSage

In this project, we will use machine learning to predict whether the S&P 500 stock index will go up or down on the following trading day. We will leverage historical price data, create relevant features, and build a predictive model.

## Introduction
The main objective of this project is to create a machine learning model that predicts whether the S&P 500 stock index's price will go up or down on a future given trading day. This prediction can be valuable for making informed investment decisions.

## Project Overview
Here's an overview of the steps involved in this project:

### Getting Started
We begin by importing the Yahoo Finance package, which allows us to access historical stock and index prices. Specifically, we initialize a ticker class for the S&P 500 index (using the symbol 'gspc') and query its historical price data. The data retrieved is stored in a Pandas DataFrame.

### Data Preparation
1. Data Cleaning and Visualization: We start by visualizing the S&P 500 price history. The data contains columns such as open, high, low, close, and volume for each trading day. We remove unnecessary columns like dividends and stock splits, which are more relevant for individual stocks than for an index.
2. Setting up the Target: The target variable is crucial for our machine learning model. In this project, our target is binary: we want to predict whether the stock price will go up or down tomorrow. To create this target variable, we first generate a new column called "tomorrow," which represents the closing price of the following day. Then, we compare today's closing price with tomorrow's closing price to determine if it's an increase or decrease. We convert this comparison into a binary target variable: 1 for an increase and 0 for a decrease.
3. Filtering Historical Data: We filter the historical data to exclude dates prior to 1990. This decision is made to ensure that the model is trained on recent data, as older data might not reflect current market conditions.

### Training the Model
With the data cleaned and the target variable prepared, we proceed to train our machine-learning model. In this project, we use a Random Forest Classifier as our initial model. The Random Forest is chosen for its ability to handle non-linear relationships in the data and resist overfitting.

### Evaluating Model Performance
1. Backtesting System: To evaluate our model's performance, we implement a backtesting system. This system allows us to simulate trading over multiple years, making predictions for each trading day. The goal is to measure how well the model performs in a real-world trading scenario.
2. Custom Threshold: We introduce a custom threshold for prediction probabilities. Instead of blindly predicting based on a 0.5 threshold, we set a higher threshold (in this case, 0.6). This approach means that the model will only predict a price increase when it is more confident, reducing the number of trades but potentially improving accuracy.

### Improving the Model
We continue to improve the model by adding additional predictors, such as rolling averages and trend indicators. These predictors provide more information for the model to consider when making predictions. The new predictors include rolling averages over various time horizons and trend columns that indicate the number of positive trading days over a specific period.

### Next Steps
The project discussed here is just a starting point for building a stock price prediction model. Here are some ideas for further improvement and extension:

* Incorporate news sentiment analysis to factor in external events and news articles that may affect stock prices.
* Include macroeconomic indicators like interest rates, inflation, and GDP growth as features for better predictions.
* Explore using higher-frequency data, such as hourly or minute-by-minute data, to make shorter-term predictions.
* Experiment with different machine learning algorithms and hyperparameter tuning to optimize model performance.

