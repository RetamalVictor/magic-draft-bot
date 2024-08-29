import pandas as pd
import requests
from time import sleep

# Function to load the dataset and rename the 'card' column to 'oracle_id'
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'card': 'oracle_id'}, inplace=True)
    return df

import requests
import time

# Function to retrieve card information from Scryfall with retry on connection error
import requests
import time

# Function to retrieve card information from Scryfall with retry on connection error
def fetch_card_info(oracle_id, retries=3, delay=5):
    url = f"https://api.scryfall.com/cards/oracle/{oracle_id}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                data=data['data'][0]
                
                if 'card_faces' in data:
                    # Handle double-faced cards
                    front_face = data['card_faces'][0]
                    back_face = data['card_faces'][1]
                    
                    # Combine oracle texts
                    combined_oracle_text = f"{front_face['oracle_text']}\n//\n{back_face['oracle_text']}"
                    
                    # Return combined information
                    return {
                        'name': data['name'],
                        'mana_cost': front_face['mana_cost'],
                        'type_line': front_face['type_line'],
                        'oracle_text': combined_oracle_text,
                        'colors': front_face.get('colors', data.get('color_identity', [])),
                        'power': front_face.get('power'),
                        'toughness': front_face.get('toughness'),
                        'loyalty': front_face.get('loyalty'),
                        # 'other_data': data  # Include the full data as a backup
                    }
                else:
                    # Handle single-faced cards
                    return data
            else:
                print(f"Failed to fetch data: Status code {response.status_code}")
                return None
        
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Unable to fetch data.")
                return None



# Function to process cards and fetch their details
def process_cards(df_to_process):
    card_details = []
    total_cards = len(df_to_process)

    for index, row in df_to_process.iterrows():
        oracle_id = row['oracle_id']
        card_info = fetch_card_info(oracle_id)

        if card_info:
            card_details.append({
                'oracle_id': oracle_id,
                'name': card_info.get('name', ''),
                'mana_cost': card_info.get('mana_cost', ''),
                'type_line': card_info.get('type_line', ''),
                'oracle_text': card_info.get('oracle_text', ''),
                'set_name': card_info.get('set_name', ''),
                'rarity': card_info.get('rarity', ''),
                'colors': ','.join(card_info.get('colors', [])),
                'power': card_info.get('power', ''),
                'toughness': card_info.get('toughness', ''),
                'loyalty': card_info.get('loyalty', ''),
                'flavor_text': card_info.get('flavor_text', ''),
                'artist': card_info.get('artist', ''),
                'illustration_id': card_info.get('illustration_id', '')
            })
        else:
            card_details.append({
                'oracle_id': oracle_id,
                'name': '',
                'mana_cost': '',
                'type_line': '',
                'oracle_text': '',
                'set_name': '',
                'rarity': '',
                'colors': '',
                'power': '',
                'toughness': '',
                'loyalty': '',
                'flavor_text': '',
                'artist': '',
                'illustration_id': ''
            })

        # Print progress
        print(f"\rProcessed {card_info.get('name', '')} ({index + 1} of {total_cards} cards, {(index + 1) / total_cards * 100:.2f}% complete)", end="", flush=True)

        # Adding a short sleep time to avoid rate-limiting
        sleep(0.1)

    return pd.DataFrame(card_details)

# Function to merge the original DataFrame with the fetched card details
def merge_and_save_data(df, card_info_df, output_path):
    updated_df = pd.merge(df, card_info_df, on='oracle_id')
    updated_df.to_csv(output_path, index=False)
    print("Data fetching and merging completed successfully.")

# Main function
if __name__ == "__main__":
    # Load and prepare the data
    data = load_and_prepare_data('data/playtest.csv')

    # Process the cards to fetch their details
    card_info_df = process_cards(data)

    # Merge and save the updated data
    merge_and_save_data(data, card_info_df, 'data/full_info_playtestv2.csv')
