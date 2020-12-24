# hacker-rank-solutions
###  statics-day0 - mean,median,mode
```ruby
size = int(input())
num = list(map(int, input().split()))
import numpy as np
from scipy import stats
print(np.mean(num))
print(np.median(num))
print(int(stats.mode(num)[0]))

```

###  statics-day0 - Weighted mean
```ruby
size = int(input())
num = list(map(int,input().split()))
wt  = list(map(int,input().split()))
res =[]
for n,w in zip(num,wt):
    res.append(n*w)
print( round(sum(res)/sum(wt),1) )
```
### 
```ruby
size = int(input())
num = list(map(int,input().split()))
num = sorted(num)
#print(num)
def getmedian(arr):
    size = len(arr)
    if size == 1:
        return arr,arr,arr[0]
    index_median = size//2
    med = 0
    if size%2 == 1:
        med = arr[size//2]
        r_array = arr[index_median+1:]
    else:
        med =(arr[size//2] + arr[((size-1)//2)])/2
        r_array = arr[index_median:]
    l_array = arr[:index_median] 
    return l_array,r_array,med
l,r,q2 = getmedian(num)
#print(l)
l1,r1,q1 = getmedian(l)
#print(r)
l3,r3,q3 = getmedian(r)
print(int(q1))
print(int(q2))
print(int(q3))
```
