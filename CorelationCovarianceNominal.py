import pandas as pd
import numpy as np

# Load data from CSV file
csv_file = "your_file.csv"
df = pd.read_csv(csv_file)
data = df.values.tolist()

# Initialize row and column sums
x, y = [], []

# Separate the data into the contingency table
r = len(data)
c = len(data[0])

table = np.array(data)

# Calculate row sums and total sum
for i in range(r):
    row_sum = sum(table[i])
    y.append(row_sum)

total_sum = sum(y)

# Calculate column sums
for j in range(c):
    col_sum = sum(table[:, j])
    x.append(col_sum)

# Calculate expected frequencies
expected = np.zeros((r, c))
for i in range(r):
    for j in range(c):
        expected[i][j] = (x[j] * y[i]) / total_sum

# Calculate the chi-square statistic
final = np.zeros((r, c))
chi = 0
for i in range(r):
    for j in range(c):
        chi_component = (((table[i][j] - expected[i][j]) ** 2) / expected[i][j])
        final[i][j] = chi_component
        chi += chi_component

print("chi^2 value:", chi)

# Use a proper threshold or compare with a p-value based on degrees of freedom
if chi > 10.8:
    print("There is correlation (Chi-Square Test)")
else:
    print("There is no correlation (Chi-Square Test)")

# Now calculate the covariance and correlation manually
n = len(df)  # Number of observations
means = df.mean().tolist()  # Calculate means of each column manually

# Initialize matrices
covariance_matrix = np.zeros((c, c))
correlation_matrix = np.zeros((c, c))

# Calculate covariance
for i in range(c):
    for j in range(c):
        covariance_sum = 0
        for k in range(n):
            covariance_sum += (df.iloc[k, i] - means[i]) * (df.iloc[k, j] - means[j])
        covariance_matrix[i][j] = covariance_sum / (n - 1)

print("\nCovariance Matrix (Manual Calculation):")
print(covariance_matrix)

# Calculate correlation
for i in range(c):
    for j in range(c):
        correlation_matrix[i][j] = covariance_matrix[i][j] / (np.sqrt(covariance_matrix[i][i] * covariance_matrix[j][j]))

print("\nCorrelation Matrix (Manual Calculation):")
print(correlation_matrix)
