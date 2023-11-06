# Capstone-Project-NaanMudlvan

**Step-by-step explanation of the Python code:**

1. **Import the necessary libraries.**
    * `pandas` for data manipulation and analysis.
    * `sklearn.model_selection` for splitting the data into training and testing sets.
    * `sklearn.linear_model` for building a linear regression model.
    * `sklearn.preprocessing` for encoding categorical variables.
    * `statsmodels.stats.outliers_influence` for calculating the variance inflation factor (VIF).

2. **Load the dataset.**
    The `read_csv()` function from Pandas is used to load the dataset into a Pandas DataFrame.

3. **Encode the categorical variable.**
    The `LabelEncoder()` class from scikit-learn is used to encode the categorical variable `vendor` into numerical values.

4. **Remove outliers.**
    The rows where the vendor has manufactured less than 5 CPUs are dropped from the dataset. This is done to avoid the model from being biased towards large vendors.

5. **Drop the unique identifier column.**
    The `model` column is dropped from the dataset as it is not needed for training the model.

6. **Split the data into training and testing sets.**
    The `train_test_split()` function from scikit-learn is used to split the data into training and testing sets. The test set size is set to 20% and the random state is set to 42 for reproducibility.

7. **Build a linear regression model.**
    The `LinearRegression()` class from scikit-learn is used to build a linear regression model.

8. **Fit the model to the training data.**
    The `fit()` method of the model is used to fit the model to the training data.

9. **Evaluate the model on the training and testing sets.**
    The `score()` method of the model is used to evaluate the model on the training and testing sets. The score is a measure of how well the model fits the data and is calculated as the R-squared value.

10. **Calculate the adjusted R-squared values.**
    The adjusted R-squared value is a penalized version of the R-squared value that takes into account the number of features in the model. It is a better measure of the model's fit when there are many features in the model.

11. **Calculate the VIF values.**
    The VIF is a measure of multicollinearity, which is the correlation between independent variables. A high VIF value indicates that the independent variables are highly correlated, which can lead to problems with the model.

12. **Predict the performance score of a new test sample.**
    The `predict()` method of the model is used to predict the performance score of a new test sample.

13. **Print the results.**
    The train score, test score, adjusted R-squared values, VIF values, and the predicted performance score for the new test sample are printed to the console.

**Conclusion:**

This Python code shows how to build and evaluate a linear regression model in Python. The code also shows how to identify and remove outliers, encode categorical variables, and calculate the adjusted R-squared and VIF values.
