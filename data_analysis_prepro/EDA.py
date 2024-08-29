import pandas as pd

# Function to load the CSV into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to display basic information about the DataFrame
def analyze_structure(df):
    print("DataFrame Information:\n")
    print(df.info())
    print("\nSummary Statistics:\n")
    print(df.describe(include='all'))
    print("\nUnique Values in Each Column:\n")
    for column in df.columns:
        unique_values = df[column].nunique()
        print(f"{column}: {unique_values} unique values")

# Function to determine which columns may need tokenization
def identify_columns_for_tokenization(df):
    text_columns = df.select_dtypes(include=['object']).columns
    potential_tokenization_columns = []
    for column in text_columns:
        unique_values = df[column].nunique()
        if unique_values > 50:  # Arbitrary threshold; adjust based on your needs
            potential_tokenization_columns.append(column)
    return potential_tokenization_columns

# Function to check for missing values and print rows with missing data
def check_missing_values(df):
    print("\nMissing Values in Each Column:\n")
    print(df.isnull().sum())
    
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    print("\nColumns with Missing Values:\n", missing_columns)
    
    # Print rows with missing values
    if missing_columns:
        missing_data_rows = df[df.isnull().any(axis=1)]
        print("\nRows with Missing Values:\n")
        print(missing_data_rows)

# Main function to run the analysis
def main():
    # Load the data
    file_path = "data/processed_playtest_datav2.csv"  # Replace with your actual file path
    data = load_data(file_path)
    
    # Analyze the structure of the DataFrame
    analyze_structure(data)
    
    # Identify columns that may need tokenization
    tokenization_columns = identify_columns_for_tokenization(data)
    print("\nColumns Recommended for Tokenization:\n", tokenization_columns)
    
    # Check for missing values and print rows with missing data
    check_missing_values(data)

if __name__ == "__main__":
    main()
