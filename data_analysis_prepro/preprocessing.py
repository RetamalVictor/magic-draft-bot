import pandas as pd
import yaml

# Function to load the CSV into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess the DataFrame according to the specified rules
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['set_name', 'flavor_text', 'artist', 'illustration_id'])
    
    # Fill NaN values according to the specified rules
    df['loyalty'] = df['loyalty'].fillna(0)
    df['colors'] = df['colors'].fillna('C')
    df['power'] = df['power'].fillna(0)
    df['toughness'] = df['toughness'].fillna(0)
    df['mana_cost'] = df['mana_cost'].fillna('0')
    df['oracle_text'] = df['oracle_text'].fillna('')
    df['rarity'] = df['rarity'].fillna('rare')
    
    # Replace any non-numeric power and toughness values containing special characters with -1
    df['power'] = df['power'].apply(lambda x: -1 if any(c in str(x) for c in ['*', 'X', '+', '-']) else int(x))
    df['toughness'] = df['toughness'].apply(lambda x: -1 if any(c in str(x) for c in ['*', 'X', '+', '-']) else int(x))
    
    # Convert data types according to specifications
    df['elo'] = df['elo'].astype(int)
    df['mainboard'] = df['mainboard'].astype(float)
    df['pickrate'] = df['pickrate'].astype(float)
    df['picks'] = df['picks'].astype(int)
    df['mainboards'] = df['mainboards'].astype(int)
    df['name'] = df['name'].astype(str)
    df['mana_cost'] = df['mana_cost'].astype(str)
    df['type_line'] = df['type_line'].astype(str)
    df['oracle_text'] = df['oracle_text'].astype(str)
    df['loyalty'] = df['loyalty'].astype(int)
    
    # Encode 'rarity' and 'colors'
    rarity_mapping = {value: idx + 1 for idx, value in enumerate(df['rarity'].unique())}
    df['rarity'] = df['rarity'].map(rarity_mapping).astype(int)
    
    colors_mapping = {value: idx + 1 for idx, value in enumerate(df['colors'].unique())}
    df['colors'] = df['colors'].map(colors_mapping).astype(int)
    
    # Save the mappings to YAML files
    with open('data/rarity_mapping.yaml', 'w') as f:
        yaml.dump(rarity_mapping, f)
    
    with open('data/colors_mapping.yaml', 'w') as f:
        yaml.dump(colors_mapping, f)
    
    return df

# Function to save the processed DataFrame to a new CSV file
def save_data(df, output_path):
    df.to_csv(output_path, index=False)

# Main function to run the preprocessing
def main():
    # Load the data
    file_path = "data/full_info_playtestv2.csv"  # Replace with your actual file path
    data = load_data(file_path)
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Save the processed data
    output_path = 'data/processed_playtest_datav2.csv'  # Replace with your desired output file path
    save_data(processed_data, output_path)

    print("Data preprocessing complete. Processed data saved to:", output_path)

if __name__ == "__main__":
    main()
