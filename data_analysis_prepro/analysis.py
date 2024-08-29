import pandas as pd

# Function to perform the Top Picked Cards analysis
def analyze_top_picked_cards(data):
    top_picked_cards = data[['name', 'pickrate', 'picks']].sort_values(by=['pickrate', 'picks'], ascending=[False, False]).head(10).reset_index()
    return top_picked_cards

# Function to perform the Top Mainboarded Cards analysis
def analyze_top_mainboarded_cards(data):
    top_mainboarded_cards = data[['name', 'mainboards']].sort_values(by=['mainboards'], ascending=[ False]).head(10).reset_index()
    return top_mainboarded_cards

# Function to perform the ELO Distribution analysis
def analyze_elo_distribution(data):
    elo_distribution = data['elo'].describe()
    return elo_distribution

# Function to perform the Rarity Impact analysis
def analyze_rarity_impact(data):
    rarity_analysis = data.groupby('rarity')[['pickrate', 'mainboard']].mean().reset_index()
    return rarity_analysis

# Function to perform the Color Distribution analysis
def analyze_color_distribution(data):
    color_distribution = data.groupby('colors')[['pickrate', 'mainboard']].mean().reset_index()
    return color_distribution

# Function to extract keywords from oracle_text
def extract_keywords(text, keywords):
    found_keywords = [keyword for keyword in keywords if keyword.lower() in str(text).lower()]
    return ', '.join(found_keywords) if found_keywords else None

# Function to extract card type from type_line
def extract_card_type(type_line):
    types = ['instant', 'sorcery', 'creature', 'enchantment', 'legendary', 'artifact', 'land', 'planeswalker']
    found_types = [t for t in types if t in str(type_line).lower()]
    return ', '.join(found_types) if found_types else None

# Function to calculate the total mana cost from mana_cost
def calculate_total_mana_cost(mana_cost):
    if pd.isna(mana_cost):
        return '0'
    total_cost = 0
    for char in mana_cost:
        if char.isdigit():
            total_cost += int(char)
        elif char.isalpha() and char != 'X':
            total_cost += 1
    return str(total_cost) if 'X' not in mana_cost else str(total_cost) + 'X'

# Function to perform the Keyword analysis
def analyze_keywords(data, keywords):
    data['keywords'] = data['oracle_text'].apply(lambda x: extract_keywords(x, keywords))
    data['card_type'] = data['type_line'].apply(extract_card_type)
    data['total_mana_cost'] = data['mana_cost'].apply(calculate_total_mana_cost)
    
    keyword_analysis = data[data['keywords'].notnull()].groupby(['keywords', 'card_type', 'total_mana_cost'])[['pickrate', 'mainboard']].mean().reset_index()
    return keyword_analysis

# Main function to execute all analyses and print results
def main(file_path:str="data/cube_list.csv"):
    # Assuming 'data' is already loaded as a pandas DataFrame
    data = pd.read_csv(file_path)

    # Perform the analyses
    top_picked_cards = analyze_top_picked_cards(data)
    top_mainboarded_cards = analyze_top_mainboarded_cards(data)
    elo_distribution = analyze_elo_distribution(data)
    rarity_analysis = analyze_rarity_impact(data)
    color_distribution = analyze_color_distribution(data)
    
    # Display the results
    print("Top Picked Cards:\n", top_picked_cards)
    print("\nTop Mainboarded Cards:\n", top_mainboarded_cards)
    print("\nELO Distribution Summary:\n", elo_distribution)
    print("\nRarity Impact Analysis:\n", rarity_analysis)
    print("\nColor Distribution Analysis:\n", color_distribution)
    
    # Define the keywords to search for
    keywords = [
        'aura', 'deathtouch', 'double strike', 'equipment', 'first strike',
        'flying', 'haste', 'lifelink', 'reach', 'trample', 'vigilance',
        'tap', 'counter spell', 'defender', 'destroy', '+1', 'infect',
        'discard', 'enchant', 'enters the battlefield', 'exile', 'flash',
        'flashback', 'goad', 'hexproof', 'indestructible',
        'legendary', 'mana', 'menace', 'graveyard battlefield',
        'player', 'the battlefield', 'sacrifice',
        'scry', 'shuffle', 'source', 'spell', 'token', 'x',
    ]
    
    keyword_analysis = analyze_keywords(data, keywords)
    
    # Display the Keyword Analysis results
    print("Keyword Analysis:\n", keyword_analysis.head())

if __name__ == "__main__":
    csv_path = "data/full_info_playtestv2.csv"
    main(csv_path)
