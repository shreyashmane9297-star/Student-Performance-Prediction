import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------
# STEP 1: Load Dataset
# -----------------------------------

data = pd.read_csv("student_data.csv")

print("Dataset loaded successfully!")


# -----------------------------------
# STEP 2: Basic Information
# -----------------------------------

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())


# -----------------------------------
# STEP 3: EDA
# -----------------------------------

print("\nStatistical Summary:")
print(data.describe())


# Study Hours vs Final Score

plt.figure(figsize=(7, 5))

sns.scatterplot(
    x="study_hours",
    y="final_score",
    data=data
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.show()


# Attendance vs Final Score

plt.figure(figsize=(7, 5))

sns.scatterplot(
    x="attendance",
    y="final_score",
    data=data
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")

plt.show()


# Previous Score vs Final Score

plt.figure(figsize=(7, 5))

sns.scatterplot(
    x="previous_score",
    y="final_score",
    data=data
)

plt.title("Previous Score vs Final Score")
plt.xlabel("Previous Exam Score")
plt.ylabel("Final Score")

plt.show()


# Correlation Heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


# -----------------------------------
# STEP 4: Separate Features and Target
# -----------------------------------

X = data.drop("final_score", axis=1)

y = data["final_score"]


print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print("final_score")


# -----------------------------------
# STEP 5: Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------
# STEP 6: Create Machine Learning Model
# -----------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# -----------------------------------
# STEP 7: Train Model
# -----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------------
# STEP 8: Make Predictions
# -----------------------------------

predictions = model.predict(X_test)


print("\nActual Final Scores:")
print(y_test.values)

print("\nPredicted Final Scores:")
print(predictions)


# -----------------------------------
# STEP 9: Evaluate Model
# -----------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n-----------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("R2 Score:", r2)

# -----------------------------------
# STEP 10: Predict New Student Score
# -----------------------------------

new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [88],
    "previous_score": [72],
    "assignment_score": [80],
    "sleep_hours": [8]
})


predicted_score = model.predict(new_student)


print("\n-----------------------------")
print("NEW STUDENT PREDICTION")
print("-----------------------------")

print(
    "Predicted Final Score:",
    round(predicted_score[0], 2)
)   


# -----------------------------------
# STEP 11: Save Trained Model
# -----------------------------------

joblib.dump(
    model,
    "student_performance_model.pkl"
)

print("\nModel saved successfully!")