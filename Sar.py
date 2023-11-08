import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoderfluence importDTimport
from statsmodels.stats.outliers_influence import variavariancence_inflation_factor

# Load the dataset and encode the 'vendor' column
data =# pd.read_csv("machine_data.csv")
data['vendor'] = LabelEncoder().fit_transform(data['vendor'])

# Remove rows for vendors with less than 5 CPUs and drop the 'model' column
data = data[data['vendor'].map(data['vendor'].value_counts()) >= 5].drop(columns='model')

# Prepare data for modeling
X, y = data.drop('score', axis=1), data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a Linear Regression model
model = LinearRegression().fit(X_train, y_train)

# Calculate train and test scores
train_score, test_score = model.score(X_train, y_train), model.score(X_test, y_test)

# Calculate adjusted R-squared values
n_train, p_train = X_train.shape[0], X_train.shape[1]
adj_r2_train = 1 - (1 - train_score) * ((n_train - 1) / (n_train - p_train - 1))

n_test, p_test = X_test.shape[0], X_test.shape[1]
adj_r2_test = 1 - (1 - test_score) * ((n_test - 1) / (n_test - p_test - 1))

# Calculate VIF values
vif = pd.DataFrame({'VIF Factor': [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
                    'features': X_train.columns})

# Predict the performance score of a new test sample
new_cpu = pd.DataFrame({'vendor': [14], 'cycle_time': [90], 'min_memory': [32], 'max_memory': [64],
                        'cache': [128], 'min_threads': [2], 'max_threads': [4]})
new_cpu_score = model.predict(new_cpu)

# Print results
print(f"Train Score: {train_score}\nTest Score: {test_score}")
print(f"Adjusted R-Squared Train: {adj_r2_train}\nAdjusted R-Squared Test: {adj_r2_test}")
print(f"VIF Values:\n{vif}\nPredicted Performance Score for the New CPU: {new_cpu_score[0]}")
