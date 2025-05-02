import pandas as pd
import os
from functools import reduce
import matplotlib.pyplot as plt

# Step 1: Load the Excel file using supported engine
file_path = "/Users/nikitakakani/Documents/Project/Dataset copy.xlsx"  
xls = pd.ExcelFile(file_path, engine="openpyxl")  

# List all available sheets in the Excel file
available_sheets = xls.sheet_names
print(f"✅ Available Sheets: {available_sheets}")

# Define and validate the selected sheets to process further
selected_sheets = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]
valid_sheets = [sheet for sheet in selected_sheets if sheet in available_sheets]

# Step 2: Define the function to clean and process individual sheet
def clean_and_process(df):
    
    # Copy the DataFrame
    df = df.copy() 

    # Strip spaces from column names
    df.columns = df.columns.str.strip() 
    
    # Standardize the primary "Date" column
    print("🔹 Standardize the primary Date column...")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Convert to datetime format
        df["Date"] = df["Date"].dt.date

    # Remove "Unnamed: 1" and "Unnamed: 2" columns
    columns_to_drop = [col for col in ["Unnamed: 1", "Unnamed: 2"] if col in df.columns]
    if columns_to_drop:
        print(f"🔹 Dropping invalid columns: {columns_to_drop}")
        df.drop(columns=columns_to_drop, inplace=True)
      
    # Handle missing or invalid data
    print("🔹 Handling missing or invalid data...")
    # Drop rows where "Date" is missing/Invalid date
    df.dropna(subset=["Date"], inplace=True)
    
    # Forward-fill missing values for other columns using the updated method
    print("🔹 Filling missing values using forward fill technique to maintain continuity in datasets...")
    df = df.ffill()

    # Print cleaned data information
    print("Cleaned Dataset format:")
    print(df.head())

    return df

# Process all valid sheets
data = {}
for sheet in valid_sheets:
    try:
        df = pd.read_excel(xls, sheet_name=sheet)
        data[sheet] = clean_and_process(df)
        print(f"✅ Successfully cleaned: {sheet}")
    except Exception as e:
        print(f"❌ Error processing sheet {sheet}: {e}")

# Save the Excel file with cleaned data
output_folder = "Cleaned_Data"
output_file = os.path.join(output_folder, "Cleaned Data File.xlsx")

# Save the cleaned data to a new Excel file
with pd.ExcelWriter(output_file) as writer:
    for sheet, df in data.items():
        df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"📁 Saved cleaned data to: {output_file}"
              f" (Sheet: {sheet})")
     
# Step 3: Merge all cleaned DataFrames
print("\n🔹 Merging all sheets into a single dataset...")

# Rename Columns to Avoid Merge Conflicts except 'Date' 
print("\n🔹 Renaming the column name in each sheet...")
for sheet, df in data.items():
    sheet_initial = sheet[0].upper() 
    df.columns = [f"{col}_{sheet_initial}" if col != "Date" else col for col in df.columns]  

# Get the list of DataFrames and ensure there are no empty dataframes
dfs = list(data.values())
dfs = [df for df in dfs if not df.empty]
print(f"Number of DataFrames to merge: {len(dfs)}")

# Use reduce to merge all DataFrames on the "Date" column
merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer", suffixes=("", "_dup")), dfs)

# Drop duplicate columns generated during the merge
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Sort the merged DataFrame by date
merged_df.sort_values(by="Date", inplace=True)

# Step 4: Align data on a daily timescale
print("\n🔹 Aligning the data on a daily timescale...")

# Ensure the dataset has a continuous date range with no gaps
min_date = merged_df["Date"].min()
max_date = merged_df["Date"].max()
date_range = pd.date_range(start=min_date, end=max_date, freq="D")
merged_df = merged_df.set_index("Date").reindex(date_range).ffill().reset_index()
merged_df.rename(columns={"index": "Date"}, inplace=True)

# Save the cleaned, merged, and aligned dataset to a single file
output_folder = "Cleaned_Data"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "Merged-Aligend Dataset.xlsx")
merged_df.to_excel(output_path, index=False)

print(f"✅ Merged and aligned data saved at: {output_path}")