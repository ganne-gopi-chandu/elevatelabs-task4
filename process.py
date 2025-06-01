import pandas as pd

# Load dataset
data = pd.read_csv("data.csv")

# Drop unnecessary columns
data.drop(columns=["id", "Unnamed: 32"], inplace=True)  # Fix added here
# Drop 'id' column (not relevant for modeling)


# Encode 'diagnosis' column (Benign -> 0, Malignant -> 1)
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

# Handle missing values (fill with column mean)
data.fillna(data.mean(), inplace=True)

# Save the processed dataset for future use
data.to_csv("processed_data.csv", index=False)

print("Data preprocessing complete! Processed dataset saved as 'processed_data.csv'.")

