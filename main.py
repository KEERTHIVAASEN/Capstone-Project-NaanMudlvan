# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Import the dataset
data = pd.read_csv("machine_data.csv")

# Step 2: Data preprocessing
# Encode the 'vendor' column using label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['vendor'] = encoder.fit_transform(data['vendor'])

# Identify vendors with less than 5 CPUs and drop corresponding rows
vendor_counts = data['vendor'].value_counts()
vendors_to_drop = vendor_counts[vendor_counts < 5].index
data = data[~data['vendor'].isin(vendors_to_drop)]

# Drop the 'model' column
data.drop('model', axis=1, inplace=True)

# Step 3: Split the data into training and testing sets
X = data.drop('score', axis=1)
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Find train and test scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Step 6: Calculate adjusted R-Squared values
n_train = X_train.shape[0]
n_features = X_train.shape[1]
adjusted_r2_train = 1 - (1 - train_score) * (n_train - 1) / (n_train - n_features - 1)
n_test = X_test.shape[0]
adjusted_r2_test = 1 - (1 - test_score) * (n_test - 1) / (n_test - n_features - 1)

# Step 7: Calculate VIF values
X_train_with_const = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Features"] = X_train_with_const.columns
vif["VIF"] = [variance_inflation_factor(X_train_with_const.values, i) for i in range(X_train_with_const.shape[1])]
vif = vif[vif["Features"] != "const"]

# Print the results
print("Train Score:", train_score)
print("Test Score:", test_score)
print("Adjusted R-Squared (Train):", adjusted_r2_train)
print("Adjusted R-Squared (Test):", adjusted_r2_test)
print("\nVIF Values:")
print(vif)

# Step 8: Predict the performance score of a new test sample
new_cpu = pd.DataFrame({
    'vendor': [14],
    'cycle_time': [90],
    'min_memory': [32],
    'max_memory': [64],
    'cache': [128],
    'min_threads': [2],
    'max_threads': [4]
})

predicted_score = model.predict(new_cpu)
print("\nPredicted Performance Score for the New CPU:", predicted_score[0])
