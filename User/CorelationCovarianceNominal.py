import numpy as np
x,y=[],[]
r=int(input("Enter no.of rows:"))
c=int(input("Enter no.of columns:"))
total_sum=()
table=np.zeros((r,c))
for i in range(r):
  print("Enter data to be entered into row",i+1,":")
  for j in range(r):
     print("Enter data of column",j+1,":")
     val=int(input())
     table[i][j]=val
for i in range(r):
    sum=()
    for j in range(c):
       sum=sum+table[i][j]
    y.append(sum)
    total_sum=total_sum+sum
for i in range(c):
   for j in range(r):
       sum=sum+table[j][i]
   x.append(sum)
   sum=0
expected=np.zeros((r,c))
for i in range(r):
   for j in range(c):
        expected[i][j]=((x[j]*y[i])/total_sum)
final=np.zeros((r,c))
for i in range(r):
   for j in range(c):
      final[i][j]=(((table[i][j]-expected[i][j])**2)/(expected[i][j]))
print(final)
chi=()
for i in range(r):
   for j in range(c):
       chi=chi+final[i][j]
print("Chi^2 value:",chi)
if(10.8<chi):
  print("There is corelation")
else:
    print("There is no corelation")
