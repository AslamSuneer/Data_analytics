import pandas as pd
import numpy as np
import statistics as st

csv_file = "co.csv"
df = pd.read_csv(csv_file)
data = df.values.tolist()
row1, row2 = [], []
for i in data:
    row1.append(i[0])
    row2.append(i[1])

row1_mean = np.mean(row1)
row2_mean = np.mean(row2)
row1_std = st.stdev(row1)
row2_std = st.stdev(row2)

correlation_matrix = np.corrcoef(row1, row2)
correlationfunction = correlation_matrix[0, 1]

covariance_matrix = np.cov(row1, row2)
covariancefunction = covariance_matrix[0, 1]

print("Correlation values using function:", correlationfunction)
print("Covariance values using function:", covariancefunction)

if correlation > 0:
    print("The correlation is positive.")
elif correlation < 0:
    print("The correlation is negative.")
else:
    print("The correlation is zero.")

