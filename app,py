import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# Assuming you have a CSV file with historical match data
data = pd.read_csv('data/football.csv')

# Feature Engineering
data['home_advantage'] = data['venue'].apply(lambda x: 1 if x == 'Home' else 0)
data['goal_difference'] = data['goals_scored'] - data['goals_conceded']

# Define features and target
features = ['home_advantage', 'goal_difference' ]  # Add more features as needed
target = 'result'  # Assuming 'result' is the target variable (Win, Lose, Draw)

X = data[features]
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prediction for a new match
new_match = pd.DataFrame({'home_advantage': [1], 'goal_difference': [2]})
predicted_result = model.predict(new_match)
print("Predicted Result:", predicted_result)