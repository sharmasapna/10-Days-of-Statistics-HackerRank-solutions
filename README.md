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
### Day 1: Quartiles
Task
Given an array, X, of  integers, calculate the respective first quartile (Q1), second quartile (Q2), and third quartile (Q3). It is guaranteed that ,Q1 ,Q2 and Q3 are integers.
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
### Day 1: Standard Deviation
```ruby
size = int(input())
arr = list(map(int, input().split(" ")))
mean = sum(arr)/size
sd_arr = []
for val in arr:
    mse = (val-mean)**2
    sd_arr.append(mse)
sd = (sum(sd_arr)/size)**0.5
sd = round(sd,1)
print(sd)
```
### Day 1: Interquartile Range


```ruby
size = int(input())
arr = list(map(int,input().split(" ")))
freq = list(map(int,input().split(" ")))
arr_final = []
for val,f in zip(arr,freq):
    for fre in range(f):
        arr_final.append(val)

num = sorted(arr_final)
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
print((q3-q1)/1.0)
```
5/6
1/9
### Day 2 
17/42
### Day3
1/3
12/51
2/3
### Day 4: Binomial Distribution I


```ruby
l = list(map(float,input().split()))
b = l[0]
g = l[1]
p_boy = b/(b+g)
p_girl = 1 - (b/(b+g))
#print(b,g)
n= 6 # total number of children
boys = 3 # at least 3 boys
p =(p_boy**3)*(p_girl**3)*20 + (p_boy**4)*(p_girl**2)*15 +(p_boy**5)*(p_girl**1)*6 + (p_boy**6)
print(round(p,3))
```



### Day 4: Binomial Distribution II

```ruby
# input
values = list(map(float, input().split()))
p = (values[0] / 100)
n = int(values[1])

def fact(n):
    if n == 1 or n==0:
        return 1
    fact = 1
    for val in range(1,n+1):
        fact *= val
    return fact

def binom(x,n,p):
    f =  fact(n)/(fact(n-x)*fact(x))
    return f * (p**x ) * (1-p)**(n-x)

# no more than 2 rejects means we have to consider {0,1,2} rejects
nm = 0 
for i in range(3):
    nm += binom(i,n,p)
print(round(nm,3))
# atleast 2 rejects means { 2,3,4,5,6,7,8,9,10}
at = 0 
for i in range(2,11):
    at += binom(i,n,p)
print(round(at,3))
    
```
### Day 4: Geometric Distribution I


```ruby
values = list(map(float, input().split()))
p = (values[0] / values[1])
n = int(input())

def gm(n,p):
    return ((1-p)**(n-1) ) * p
print(round( gm(n,p),3) )
```
### Day 4: Geometric Distribution II
```ruby
values = list(map(float, input().split()))
p = (values[0] / values[1])
n = int(input())

def gm(n,p):
    return ((1-p)**(n-1) ) * p
res = 0
for i in range (1,n+1):
    res += gm(i,p)
print(round(res,3))

```
### Day 5: Poisson Distribution I


```ruby
import math
# input
l = float(input())
k = int(input())
def fact(n):
    if n == 0 or n == 1:
        return 1
    res = 1
    for val in range(1,n+1):
        res = res*val
    return res
#print(fact(5))
def pois(k,lmda):
    p = (lmda**(k) * math.exp(-lmda)) / fact(k)
    return(p)
print(round(pois(k,l),3))
```
### Day 5: Poisson Distribution II
```ruby
import math
# input
means = list(map(float,input().split(" ")))
mean_a, mean_b = float(means[0]), float(means[1])
# factorial
def fact(n):
    if n == 0 or n == 1:
        return 1
    res = 1
    for val in range(1,n+1):
        res = res*val
    return res
#print(fact(5))
# poison calculation
def pois(k,lmda):
    p = (lmda**(k) * math.exp(-lmda)) / fact(k)
    return(p)
# calculating the expected value
exp_a=0
exp_b=0
for val in range(100):
    exp_a += (160 + 40 * (val**2)) * pois(val,mean_a)
    exp_b += (128 + 40 * (val**2)) * pois(val,mean_b)

print(round(exp_a,3))
print(round(exp_b,3))
```


