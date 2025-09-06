import pandas as pd

# Load the dataset
df = pd.read_csv('blinkit_sales_data.csv')

# Handle missing values
# Item_Weight: Replace missing values with the mean
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

# Outlet_Size: Replace missing values with the mode
df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

# Feature Engineering
# Create a new feature for 'Years_Established'
df['Years_Established'] = 2025 - df['Outlet_Establishment_Year']

# Item_Visibility: Replace 0 with mean
df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())

# Item_Fat_Content: Standardize categories
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

# Drop original columns that are no longer needed
df.drop(['Outlet_Establishment_Year'], axis=1, inplace=True)

# Save the preprocessed data
df.to_csv('blinkit_preprocessed_data.csv', index=False)

print('Data preprocessing and feature engineering complete. Saved to blinkit_preprocessed_data.csv')

