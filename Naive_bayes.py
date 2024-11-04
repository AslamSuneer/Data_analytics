# Step 1: Read CSV Data
def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            row = line.strip().split(',')
            features = row[1:-1]  # Age, income, student, credit_rating
            label = row[-1]       # buys_computer
            data.append((features, label))
    return data

# Step 2: Calculate Prior Probabilities P(class)
def calculate_prior_probabilities(data):
    total_samples = len(data)
    class_counts = {}

    for features, label in data:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    priors = {}
    for label in class_counts:
        priors[label] = class_counts[label] / total_samples

    return priors

# Step 3: Calculate Likelihood P(feature|class) with Laplace Smoothing
def calculate_likelihoods(data, alpha=1):  # Laplace smoothing parameter alpha=1
    feature_counts = {}
    class_counts = {}

    for features, label in data:
        if label not in class_counts:
            class_counts[label] = 0
            feature_counts[label] = [{} for _ in range(len(features))]
        class_counts[label] += 1
        for i in range(len(features)):
            if features[i] not in feature_counts[label][i]:
                feature_counts[label][i][features[i]] = 0
            feature_counts[label][i][features[i]] += 1

    likelihoods = {}
    for label in feature_counts:
        likelihoods[label] = []
        for i in range(len(feature_counts[label])):
            total_feature_count = sum(feature_counts[label][i].values())
            unique_values = len(feature_counts[label][i])  # For Laplace smoothing
            likelihoods[label].append({key: (feature_counts[label][i][key] + alpha) / 
                                       (total_feature_count + alpha * unique_values)
                                       for key in feature_counts[label][i]})

    return likelihoods

# Step 4: Classify new data and print probabilities
def classify(priors, likelihoods, new_data, alpha=1):
    posteriors = {}
    for label in priors:
        posteriors[label] = priors[label]  # Start with the prior probability
        print(f"\nClass: {label}")
        print(f"Prior probability: {priors[label]:.4f}")
        
        for i in range(len(new_data)):
            feature_value = new_data[i]
            feature_likelihoods = likelihoods[label][i]
            # Calculate total feature count with smoothing
            unique_values = len(feature_likelihoods)
            total_feature_count = sum(feature_likelihoods.values())
            # Use Laplace smoothing if feature is unseen
            feature_probability = feature_likelihoods.get(
                feature_value,
                alpha / (total_feature_count + alpha * unique_values)
            )
            print(f"P({feature_value}|{label}) = {feature_probability:.4f}")
            
            posteriors[label] *= feature_probability

        print(f"Posterior probability for {label}: {posteriors[label]:.4f}")

    # Return the class label with the highest posterior probability
    return max(posteriors, key=posteriors.get)

# Step 5: Running the classifier on new data
def naive_bayes_classifier(training_data, new_data):
    priors = calculate_prior_probabilities(training_data)
    likelihoods = calculate_likelihoods(training_data)
    return classify(priors, likelihoods, new_data)

# Main execution
filename = 'naive.csv'  # Name of your CSV file
training_data = read_csv_file(filename)

# Classify the new data: X = (age=youth, income=medium, student=yes, credit_rating=fair)
new_sample = ['youth', 'medium', 'yes', 'fair']
predicted_class = naive_bayes_classifier(training_data, new_sample)
print(f'\nPredicted class for {new_sample}: {predicted_class}')
