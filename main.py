import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

selected_features = ['nlipktrlsm', 'llsketrlsm', 'toefltrlsm', 'stwsdtrlsm', 'jalur']
X = train_data[selected_features]
y = train_data['tepat_waktu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisikan model Decision Tree
model = DecisionTreeClassifier()

# Tentukan ruang hyperparameter yang akan dieksplorasi
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
test_predictions = best_model.predict(test_data[selected_features])

submission_df = pd.DataFrame({
    'index': test_data['index'],
    'tepat_waktu': test_predictions
})

submission_df.head()

submission_df.to_csv('fourth-submission.csv', index=False)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
