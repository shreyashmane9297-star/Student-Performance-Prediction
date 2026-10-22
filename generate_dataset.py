import pandas as pd
import numpy as np


# -----------------------------------
# Generate Student Dataset
# -----------------------------------

np.random.seed(42)

number_of_students = 500


# -----------------------------------
# Generate Input Features
# -----------------------------------

study_hours = np.random.uniform(
    1, 10, number_of_students
)

attendance = np.random.uniform(
    50, 100, number_of_students
)

previous_score = np.random.uniform(
    40, 95, number_of_students
)

assignment_score = np.random.uniform(
    40, 100, number_of_students
)

sleep_hours = np.random.uniform(
    5, 10, number_of_students
)


# -----------------------------------
# Calculate Final Score
# -----------------------------------

final_score = (
    study_hours * 2.5
    + attendance * 0.20
    + previous_score * 0.35
    + assignment_score * 0.15
    + sleep_hours * 1.0
)


# Add natural variation
noise = np.random.normal(
    0, 3, number_of_students
)

final_score = final_score + noise


# Keep score between 0 and 100
final_score = np.clip(
    final_score,
    0,
    100
)


# -----------------------------------
# Create DataFrame
# -----------------------------------

data = pd.DataFrame({
    "study_hours": study_hours.round(1),
    "attendance": attendance.round(1),
    "previous_score": previous_score.round(1),
    "assignment_score": assignment_score.round(1),
    "sleep_hours": sleep_hours.round(1),
    "final_score": final_score.round(1)
})


# -----------------------------------
# Save Dataset
# -----------------------------------

data.to_csv(
    "student_data.csv",
    index=False
)


print("Dataset created successfully!")

print("\nNumber of students:", len(data))

print("\nFirst 5 records:")
print(data.head())

print("\nDataset saved as student_data.csv")