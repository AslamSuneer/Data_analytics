import csv
import math

def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            row = line.strip().split(',')
            features = row[1:-1]  # All but the last column
            label = row[-1]  # Last column is the label
            data.append((features, label))
    return data

def entropy(data):
    total_samples = len(data)
    label_counts = {}
    for features, label in data:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    ent = 0
    for label in label_counts:
        prob = label_counts[label] / total_samples
        ent -= prob * math.log2(prob)
    return ent

def split_data(data, index, value):
    true_branch = [row for row in data if row[0][index] == value]
    false_branch = [row for row in data if row[0][index] != value]
    return true_branch, false_branch

def find_best_split(data):
    best_gain = 0
    best_index = None
    best_value = None
    current_entropy = entropy(data)
    n_features = len(data[0][0])  # Number of features
    for index in range(n_features):
        values = set(row[0][index] for row in data)  # Unique values for the feature
        for value in values:
            true_branch, false_branch = split_data(data, index, value)
            if not true_branch or not false_branch:
                continue
            p = len(true_branch) / len(data)
            gain = current_entropy - p * entropy(true_branch) - (1 - p) * entropy(false_branch)
            if gain > best_gain:
                best_gain, best_index, best_value = gain, index, value
    return best_gain, best_index, best_value

class DecisionNode:
    def __init__(self, index=None, value=None, true_branch=None, false_branch=None, prediction=None):
        self.index = index
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.prediction = prediction

def build_tree(data):
    gain, index, value = find_best_split(data)
    if gain == 0:
        # No more gain, return leaf node
        return DecisionNode(prediction=data[0][1])
    true_branch, false_branch = split_data(data, index, value)
    true_node = build_tree(true_branch)
    false_node = build_tree(false_branch)
    return DecisionNode(index=index, value=value, true_branch=true_node, false_branch=false_node)

def print_tree(node, headers, spacing=""):
    if node.prediction is not None:
        print(spacing + f"Predict: {node.prediction}")
        return
    print(spacing + f"{headers[node.index]} == {node.value}?")
    print(spacing + '-> True:')
    print_tree(node.true_branch, headers, spacing + "    ")
    print(spacing + '-> False:')
    print_tree(node.false_branch, headers, spacing + "    ")

def classify(tree, sample):
    if tree.prediction is not None:
        return tree.prediction
    if sample[tree.index] == tree.value:
        return classify(tree.true_branch, sample)
    else:
        return classify(tree.false_branch, sample)

def get_user_input(headers):
    user_input = []
    for header in headers:
        value = input(f"Enter value for {header}: ")
        user_input.append(value)
    return user_input

if __name__ == "__main__":
    filename = 'naive.csv'  # Ensure this file exists in the same directory
    headers = ['age', 'income', 'student', 'credit_rating']
    training_data = read_csv_file(filename)

    tree = build_tree(training_data)

    print("Decision Tree Structure:")
    print_tree(tree, headers)

    new_sample = get_user_input(headers)
    predicted_class = classify(tree, new_sample)
    print(f"\nPredicted class for {new_sample}: {predicted_class}")
