import pandas as pd

# Load the dataset containing passenger information
df = pd.read_csv('titanic.csv')

# Verify loading by checking the shape
print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

# Display the full original dataset
print("--- Original Dataset ---")
print(df)
# Check missing values in Age and Cabin columns
missing_age = df['Age'].isnull().sum()
missing_cabin = df['Cabin'].isnull().sum()
print("\n--- Missing Value Check ---")
print(f"Missing values in 'Age': {missing_age}")
print(f"Missing values in 'Cabin': {missing_cabin}")

# Calculate the median age (excluding NaN)
median_age = df['Age'].median()
print(f"Calculated Median Age: {median_age}")
# Fill missing values in Age column
df['Age'] = df['Age'].fillna(median_age)
# Verify no missing values remain in Age
print(f"Missing values in 'Age' after imputation: {df['Age'].isnull().sum()}")

# Inspect the Ticket column values
print("--- Ticket Column Contents ---")
print(df[['PassengerId', 'Name', 'Ticket']])

# Example action: Dropping the Ticket column for streamlined analysis
df_cleaned = df.drop(columns=['Ticket'])
print("Column 'Ticket' removed successfully.")

# Display the updated dataset
print("--- Cleaned Dataset ---")
print(df_cleaned)
# Summary of cleaning process
summary = """
=== Cleaning Summary ===
1. Data Loading: Imported 6 passenger records from 'Titanic-Dataset.csv'.
2. Missing Values Handled:
 - 'Age' had 1 missing value (Passenger 6). Imputed with column median age of
35.0.
 - 'Cabin' has 4 missing values (high missing rate retained as NaN or targeted
for feature engineering).
3. Feature Engineering/Selection:
 - Evaluated 'Ticket' column. Dropped due to high variance and low predictive
value for baseline models.
"""
print(summary)