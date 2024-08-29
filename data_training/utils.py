import torch
from collections import Counter
import re


def parse_mana_cost(mana_cost):
    """Extracts the number and letters from the mana cost string."""
    # Use a regular expression to extract the contents inside the curly braces
    matches = re.findall(r'\{(\d+|[A-Za-z]+)\}', mana_cost)
    
    # Separate numbers and letters
    num = 0
    letters = ''
    
    for match in matches:
        if match.isdigit():
            num += int(match)  # Sum up all numbers
        else:
            letters += match  # Concatenate all letter components

    return num, letters

def letter_similarity(letters1, letters2):
    """Computes a similarity score based on the overlap of letters."""
    count1 = Counter(letters1)
    count2 = Counter(letters2)
    intersection = sum((count1 & count2).values())
    union = sum((count1 | count2).values())
    return intersection / union if union > 0 else 0

def prepare_training_data(df, n_negatives=10, n_half_pos=10):
    training_data = []
    for _, row in df.iterrows():
        mana_token = torch.tensor([row['mana_token']])
        type_token = torch.tensor([row['type_token']])

        # Select positive examples (same mana_cost, same type_line)
        positive_examples = df[(df['mana_cost'] == row['mana_cost']) & (df['type_line'] == row['type_line'])]
        if len(positive_examples) > 1:
            pos_row = positive_examples.sample().iloc[0]
            pos_token = torch.tensor([pos_row['mana_token'], pos_row['type_token']])
        else:
            continue

        # Calculate similarity scores for negative examples
        target_mana_cost = row['mana_cost']
        negative_examples = df[(df['mana_cost'] != row['mana_cost']) | (df['type_line'] != row['type_line'])].copy()
        half_pos_examples = df[(df['type_line'] == row['type_line']) & (df['mana_cost'] != row['mana_cost'])].copy()
        if half_pos_examples.empty:
            # Reverse conditions: different type_line and same mana_cost
            half_pos_examples = df[(df['type_line'] != row['type_line']) & (df['mana_cost'] == row['mana_cost'])].copy()

        def compute_similarity(row):
            num1, letters1 = parse_mana_cost(target_mana_cost)
            num2, letters2 = parse_mana_cost(row['mana_cost'])
            num_diff = abs(num1 - num2)
            letter_diff = abs(len(letters1) - len(letters2))
            similarity_score = letter_similarity(letters1, letters2)
            total_diff = num_diff + letter_diff - similarity_score
            return total_diff

        negative_examples['similarity_score'] = negative_examples.apply(compute_similarity, axis=1)
        negative_examples = negative_examples.sort_values(by='similarity_score', ascending=False).head(n_negatives)
        for _, neg_row in negative_examples.iterrows():
            neg_token = torch.tensor([neg_row['mana_token'], neg_row['type_token']])
            training_data.append((mana_token, type_token, pos_token, neg_token))

        half_pos_examples['similarity_score'] = half_pos_examples.apply(compute_similarity, axis=1)
        half_pos_examples = half_pos_examples.sort_values(by='similarity_score', ascending=True).head(n_half_pos)
        for _, half_pos_row in half_pos_examples.iterrows():
            half_pos_token = torch.tensor([half_pos_row['mana_token'], half_pos_row['type_token']])
            neg_row = df[(df['mana_cost'] != row['mana_cost']) | (df['type_line'] != row['type_line'])].sample().iloc[0]
            neg_token = torch.tensor([neg_row['mana_token'], neg_row['type_token']])
            training_data.append((mana_token, type_token, pos_token, half_pos_token, neg_token))

    
    return training_data
