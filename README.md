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
