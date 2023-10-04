import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the training data and test data
training_data_path = "data_training.csv"
test_data_path = "data_test.csv"
training_data = pd.read_csv(training_data_path)
test_data = pd.read_csv(test_data_path)

# Convert categorical variables to dummy variables for both training and test datasets
categorical_features = ["Feature_2", "Feature_4", "Feature_5", "Feature_7", "Feature_6"]
training_data = pd.get_dummies(training_data, columns=categorical_features)
test_data = pd.get_dummies(test_data, columns=categorical_features)


# print(training_data.columns)

# Separate the features and target label for the training dataset
X_train = training_data.drop(columns=["Label"])
y_train = training_data["Label"]


# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define a grid of hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best model to predict the labels for the test dataset
best_rf_classifier = grid_search.best_estimator_
test_predictions = best_rf_classifier.predict(test_data)

# Save the predictions to a CSV file
output_path = "P1_test_output.csv"
pd.DataFrame(test_predictions).to_csv(output_path, header=False, index=False)


# 1, load the training data and test data

# 2, convert categorical variable to dummy variable for training data and test data. 

# 3, use cross validation to split the training data into 2 parts one for trainin and one for testing to select the best model.  
# random forest  (cross_val_score, use f1 score as evaluation critera, gridsearchcv)

# 4, tune the model hyperparameter using gridsearchcv

# 5, generate .csv for the test set