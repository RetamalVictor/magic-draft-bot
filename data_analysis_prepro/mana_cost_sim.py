import itertools
from collections import Counter

# Load the mana costs from the previously generated JSON file
import json

with open('data/mana_costs.json', 'r') as f:
    mana_data = json.load(f)
    all_mana_costs = mana_data['mana_costs']

def parse_mana_cost(mana_cost):
    """Extracts the number and letters from the mana cost."""
    parts = mana_cost.split(',')
    num = int(parts[0])
    letters = ''.join(parts[1:])
    return num, letters

def letter_similarity(letters1, letters2):
    """Computes a similarity score based on the overlap of letters."""
    count1 = Counter(letters1)
    count2 = Counter(letters2)
    intersection = sum((count1 & count2).values())
    union = sum((count1 | count2).values())
    return intersection / union if union > 0 else 0

def retrieve_closest_mana_costs(target_mana_cost, all_mana_costs, n=15):
    target_num, target_letters = parse_mana_cost(target_mana_cost)
    candidates = []

    for mana_cost in all_mana_costs:
        if mana_cost == target_mana_cost:
            continue  
        num, letters = parse_mana_cost(mana_cost)
        num_diff = abs(num - target_num)
        letter_diff = abs(len(letters) - len(target_letters))
        similarity_score = letter_similarity(letters, target_letters)
        
        # We prioritize number and length similarity, then add similarity score
        total_diff = num_diff + letter_diff - similarity_score
        candidates.append((total_diff, mana_cost))

    # Sort by the computed score and get the top n
    candidates.sort(key=lambda x: x[0])
    return [mana_cost for _, mana_cost in candidates[:n]]

# Example usage
target = "3,B,U"
closest_mana_costs = retrieve_closest_mana_costs(target, all_mana_costs)
print(closest_mana_costs)
