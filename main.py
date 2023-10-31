import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import the dataset
data = pd.read_csv("machine_data.csv")

# Encode the 'vendor' column using label encoder
le = LabelEncoder()
data['vendor'] = le.fit_transform(data['vendor'])

# Identify vendors who have manufactured less than 5 CPUs and drop those rows
vendor_counts = data['vendor'].value_counts()
data = data[~data['vendor'].isin(vendor_counts[vendor_counts < 5].index)]

# Drop the 'model' column
data.drop('model', axis=1, inplace=True)

# Select 'score' as the target variable and remaining features as predictors
X = data.drop('score', axis=1)
y = data['score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Find the train and test scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Calculate the adjusted R-Squared values
n_train = X_train.shape[0]
p_train = X_train.shape[1]
adj_r2_train = 1 - (1 - train_score) * ((n_train - 1) / (n_train - p_train - 1))

n_test = X_test.shape[0]
p_test = X_test.shape[1]
adj_r2_test = 1 - (1 - test_score) * ((n_test - 1) / (n_test - p_test - 1))

# Calculate the VIF values
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns

# Predict the performance score of a new test sample
new_cpu = pd.DataFrame({
    'vendor': [14],
    'cycle_time': [90],
    'min_memory': [32],
    'max_memory': [64],
    'cache': [128],
    'min_threads': [2],
    'max_threads': [4]
})
new_cpu_score = model.predict(new_cpu)

print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")
print(f"Adjusted R-Squared Train: {adj_r2_train}")
print(f"Adjusted R-Squared Test: {adj_r2_test}")
print(f"VIF Values: \n{vif}")
print(f"\nPredicted Performance Score for the New CPU: {new_cpu_score[0]}")