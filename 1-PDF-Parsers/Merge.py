import pandas as pd

# Paths to the CSV files
file1 = 'file1.csv'
file2 = 'file2.csv'

# Read the two CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the dataframes. Here, we simply concatenate them.
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_output.csv', index=False)

print("Merged CSV file has been saved as 'merged_output.csv'.")

