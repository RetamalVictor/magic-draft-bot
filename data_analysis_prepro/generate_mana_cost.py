import itertools
import json

# Define the range of numbers and the letters
numbers = list(range(15))  # 0 to 14
letters = ['B', 'U', 'G', 'W', 'R', 'C']

# Generate all combinations
mana_costs = []

for num in numbers:
    for r in range(6):  # 0 to 5 letters
        for combo in itertools.combinations_with_replacement(letters, r):
            if r == 0:  # No letters
                mana_cost = f"{num}"
            else:
                mana_cost = f"{num},{','.join(combo)}"
            mana_costs.append(mana_cost)

# Save the result to a JSON file
output = {"mana_costs": mana_costs}

with open('data/mana_costs.json', 'w') as f:
    json.dump(output, f, indent=4)

print("Mana costs generated and saved to mana_costs.json")
