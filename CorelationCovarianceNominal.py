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
    print("There is correlation")
else:
    print("There is no correlation")
