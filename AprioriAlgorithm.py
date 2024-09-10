import csv
from itertools import combinations

# Load the transactions from CSV file
file_path = "apr.csv"
transactions = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    transactions = [list(row) for row in reader]

# Apriori algorithm to find frequent itemsets
def apriori(transactions, min_support):
    c1 = {}
    for transaction in transactions:
        for item in transaction:
            if item in c1:
                c1[item] += 1
            else:
                c1[item] = 1

    # Generate L1 by filtering items that meet min_support
    l1 = {key: value for key, value in c1.items() if value / len(transactions) >= min_support}
    l = [l1]  # List of dictionaries to hold frequent itemsets
    k = 2

    while len(l[k-2]) > 0:
        ck = {}
        for transaction in transactions:
            combos = combinations(transaction, k)
            for combo in combos:
                if combo in ck:
                    ck[combo] += 1
                else:
                    ck[combo] = 1

        # Generate Lk by filtering candidates that meet min_support
        lk = {key: value for key, value in ck.items() if value / len(transactions) >= min_support}
        if len(lk) == 0:
            break
        l.append(lk)
        k += 1

    # Flatten frequent itemsets and return the keys
    return [item for sublist in l for item in sublist.keys()]

# Function to generate association rules from frequent itemsets
def association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        for i in range(1, len(itemset)):
            antecedents = [x for x in combinations(itemset, i)]
            for antecedent in antecedents:
                consequent = tuple([item for item in itemset if item not in antecedent])
                antecedent_support = sum([1 for transaction in transactions if set(antecedent).issubset(set(transaction))])
                both_support = sum([1 for transaction in transactions if set(antecedent + consequent).issubset(set(transaction))])
                
                try:
                    confidence = both_support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent))
                except ZeroDivisionError:
                    pass
    return rules

# Set minimum support and confidence thresholds
min_support = 2 / len(transactions)
frequent_itemsets = apriori(transactions, min_support)
min_confidence = 0.75

# Generate and print association rules
rules = association_rules(frequent_itemsets, transactions, min_confidence)
for rule in rules:
    print(f"{rule[0]} => {rule[1]}")
