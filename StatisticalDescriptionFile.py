import numpy as np
import pandas as pd

df = pd.read_csv('statistical.csv')
print(df)
age = df[" marks"]
def mode(ls):
	dict = {}
	for i in ls:
		if i not in dict:
			dict[i] = 0
		dict[i] += 1
		
	max = 0
	for i in dict:
		if dict[i] > max:
			max = dict[i]
	list1 = []
	for i in dict:
		if dict[i] == max:
			list1.append(i)
	return list1


		
print('Mean:', np.mean(age))
print('Median:', np.median(age))
print('Mode:', mode(age))
modename = {0:'no mode',1:'unimode',2:'bimode',3:'trimode'}
mode_list = mode(age)
if len(mode_list) > 0 and len(mode_list) < 4:
	print(modename[len(mode(age))])
else:
	print("neither mode")
def quantile(ls):
	ls = sorted(ls)
	quantiles = [ls[0]]
	for i in range(1, 4):
		value = int(len(ls) * i/4)
		quantiles.append(ls[value])
	quantiles.append(ls[len(ls)-1])
	return quantiles

quantile_list = quantile(age)
iqr = quantile_list[3] - quantile_list[1]
higher_outlier = quantile_list[3] + 1.5*iqr
lower_outlier = quantile_list[1] - 1.5*iqr
new_ls = []
for i in age:
	if i <= higher_outlier and i >= lower_outlier:
		new_ls.append(i)
		
print('Quantiles:',quantile_list)
print('Higher outlier:', higher_outlier)
print('Lower outlier:', lower_outlier)
print('outlier removed list:',new_ls)
print('Standard deviation:', np.std(new_ls))
