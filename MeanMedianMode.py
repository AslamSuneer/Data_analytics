l=[]
n=int(input("Enter total no of elements:"))
print("Enter the elements:")
for i in range(0,n):
              s=int(input())
              l.append(s)
l.sort()
mean=sum(l)
mean=mean/len(l)
print("mean: ",mean)
if(n%2==0):
     median=(l[int(n/2-1)]+l[int(n/2)])/2
     print("median: ",median)
else:
    size=(n+1)/2-1
    print("median: ",l[int(size)])
i=0
max=-1
a=-1
while(i<n):
      count=1
      j=i+1
      while(j<n and l[j]==l[i]):
            count+=1
            j+=1
      if(count>max):
            max=count
            a=l[i]
      i=j
print("mode: ",a)
