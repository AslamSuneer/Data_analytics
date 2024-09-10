import pandas as pd
df=pd.read_csv('cono.csv')
print(df)
print()
print()

from scipy.stats import chi2_contingency
df=pd.read_csv('cono.csv')
contingency_table=pd.crosstab(df['a'],df['b'])
print(contingency_table)
print()
print()

chi2,p,dof,expected=chi2_contingency(contingency_table)
print("Chi2_statistic:",chi2)
print("pvalue:",p)
print("dof:",dof)
print("expected frquency:",expected)
alpha=0.05
if p<alpha:
   print("There is significant realation between a and b")
else:
   print("There is no significant realation between a and b")
