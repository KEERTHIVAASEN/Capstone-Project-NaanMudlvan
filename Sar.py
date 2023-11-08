import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['vendor'] = LabelEncoder().fit_transform(data['vendor'])
    data = data[data['vendor'].map(data['vendor'].value_counts()) >= 5].drop(columns='model')
    return data

def split_data(data):
    X, y = data.drop('score', axis=1), data['score']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    return LinearRegression().fit(X_train, y_train)

def calculate_scores(model, X_train, X_test, y_train, y_test):
    train_score, test_score = model.score(X_train, y_train), model.score(X_test, y_test)
    return train_score, test_score

def calculate_adjusted_r2(n, p, score):
    return 1 - (1 - score) * ((n - 1) / (n - p - 1))

def calculate_vif(X_train):
    return pd.DataFrame({'VIF Factor': [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
                         'features': X_train.columns})

def predict_new_cpu_score(model, new_cpu):
    return model.predict(new_cpu)

# Load and preprocess data
data = load_and_preprocess_data("machine_data.csv")

# Split data
X_train, X_test, y_train, y_test = split_data(data)

# Train model
model = train_model(X_train, y_train)

# Calculate scores
train_score, test_score = calculate_scores(model, X_train, X_test, y_train, y_test)

# Calculate adjusted R-squared values
adj_r2_train = calculate_adjusted_r2(X_train.shape[0], X_train.shape[1], train_score)
adj_r2_test = calculate_adjusted_r2(X_test.shape[0], X_test.shape[1], test_score)

# Calculate VIF values
vif = calculate_vif(X_train)

# Predict the performance score of a new test sample
new_cpu = pd.DataFrame({'vendor': [14], 'cycle_time': [90], 'min_memory': [32], 'max_memory': [64],
                        'cache': [128], 'min_threads': [2], 'max_threads': [4]})
new_cpu_score = predict_new_cpu_score(model, new_cpu)

# Print results
print(f"Train Score: {train_score}\nTest Score: {test_score}")
print(f"Adjusted R-Squared Train: {adj_r2_train}\nAdjusted R-Squared Test: {adj_r2_test}")
print(f"VIF Values:\n{vif}\nPredicted Performance Score for the New CPU: {new_cpu_score[0]}")
