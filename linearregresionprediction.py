import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Database Connection
# Example: PostgreSQL
engine = create_engine("postgresql://username:password@localhost:5432/your_database")

# Step 2: Load leads data from the database
query = "SELECT product, source, budget, follow_ups, decision_maker, timeline_days, converted FROM leads"
data = pd.read_sql(query, engine)

# Step 3: Define features and target
X = data.drop('converted', axis=1)
y = data['converted']

# Step 4: Preprocessing pipeline
categorical_features = ['product', 'source']
numeric_features = ['budget', 'follow_ups', 'decision_maker', 'timeline_days']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression(solver='liblinear'))  # 'liblinear' is stable for small datasets
])

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
pipeline.fit(X_train, y_train)

# Step 7: Create a new lead (e.g., from user input or another DB call)
new_lead = pd.DataFrame([{
    'product': 'A',
    'source': 'Referral',
    'budget': 50000,
    'follow_ups': 3,
    'decision_maker': 1,
    'timeline_days': 15
}])

# Step 8: Predict probability
prob = pipeline.predict_proba(new_lead)[0][1]
score = int(prob * 100)

# Step 9: Output
print(f"Predicted probability of conversion: {prob:.2f}")
print(f"Lead score (0–100): {score}")
