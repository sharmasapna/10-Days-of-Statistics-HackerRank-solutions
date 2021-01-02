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
### Day 5: Normal Distribution I
```ruby
import math
# input
nd = list(input().split(" "))
mean = float(nd[0])
var  = float(nd[1])**2
less_than = float(input())
range_ = list(input().split(" "))
from_val = float(range_[0])
to_val   = float(range_[1])


def cum_pdf(x,mean,var):
    return 0.5*(1 + math.erf( (x-mean)/(2*var)**(0.5) ))


# less than 19.5 hours

p1 = cum_pdf(less_than,mean,var)

# between 20 and 22

p2 = cum_pdf(to_val,mean,var) - cum_pdf(from_val,mean,var) 
print(round(p1,3))
print(round(p2,3))
```

### Day 5: Normal Distribution II
```ruby
import math
# input
nd = list(input().split(" "))
mean = float(nd[0])
var  = float(nd[1])**2
more_than = int(input())
less_than = int(input()) 


def cum_pdf(x,mean,var):
    return 0.5*(1 + math.erf( (x-mean)/(2*var)**(0.5) ))

p1 = cum_pdf(more_than,mean,var)
p1 = 1-p1
p2 = cum_pdf(less_than,mean,var) 
p2 = 1-p2
p3 = cum_pdf(less_than,mean,var) 
print(round(p1*100,2))
print(round(p2*100,2))
print(round(p3*100,2))
```
### Day 6: The Central Limit Theorem I
```ruby
import math
max_wt = float(input())
num_boxes = float(input())
mean = float(input())
sd = float(input())
var = sd**2
mean_prime = num_boxes * mean
sd_prime = (num_boxes**(0.5)) * sd
var_prime = num_boxes * (sd**2)


def cum_pdf(x,mean,var):
    return 0.5*(1 + math.erf( (x-mean)/(2*var)**(0.5) ))
p = cum_pdf(max_wt,mean_prime,var_prime)
print(round(p,4))
```
### Day 6: The Central Limit Theorem II
```ruby
import math
last_min = float(input())
num_students = float(input())
mean = float(input())
sd = float(input())
var = sd**2
mean_prime = num_students * mean
sd_prime = (num_students**(0.5)) * sd
var_prime = num_students * (sd**2)


def cum_pdf(x,mean,var):
    return 0.5*(1 + math.erf( (x-mean)/(2*var)**(0.5) ))
p = cum_pdf(last_min,mean_prime,var_prime)
print(round(p,4))
```
### Day 6: The Central Limit Theorem III
```ruby
import math
sample_size = float(input())
mean = float(input())
sd = float(input())
per_cover = float(input())
z = float(input())

a = mean - z * (sd / math.sqrt(sample_size))
b = mean + z * (sd / math.sqrt(sample_size))

print (round(a,2))
print (round(b,2))
```
### Day 7: Pearson Correlation Coefficient I


```ruby
import statistics
size = int(input())
x = list(map(float,input().split()))
y = list(map(float,input().split()))
mean_x,mean_y = statistics.mean(x) , statistics.mean(y)
sd_x,sd_y = statistics.pstdev(x),statistics.pstdev(y)
num =0
for a,b in zip(x,y):
    num += (a-mean_x)*(b-mean_y)
pcc = num/(size*sd_x*sd_y)
print(round(pcc,3))
```
### Day 7: Spearman's Rank Correlation Coefficient
```ruby
size = int(input())
x = list(map(float,input().split()))
y = list(map(float,input().split()))

rank_x = []
for val in x:
    rank = sorted(x).index(val) +1
    rank_x.append(rank)

rank_y = []
for val in y:
    rank = sorted(y).index(val) +1
    rank_y.append(rank)

d=0
for a,b in zip(rank_x,rank_y):
    d += (a-b)**2
srcc = 1 - 6*d/((size**2-1)*size)
    
print(round(srcc,3))

```
### Day 8: Least Square Regression Line
```ruby
x = [95,85,80,70,60]
y = [85,95,70,65,70]
mean_x,mean_y = sum(x)/len(x), sum(y)/len(y)
x_square = 0
xy = 0
for a,b in zip(x,y):
    x_square += a**2
    xy += a*b
n = len(x)
# calculating the slope
b = ( n*xy - sum(x)*sum(y)) / (n*x_square - (sum(x))**2 )

# calculating the intercept
a = mean_y - b*mean_x

y_pred  = a + b* 80
print(round(y_pred,3))
```
### Day 8: Pearson Correlation Coefficient II
-3/4

### Day 9: Multiple Linear Regression
```ruby
from sklearn import linear_model
# input
i = input().split() 
n = int(i[0])       # number of features    
l = int(i[1])       # number of samples
x=[]
y=[]
for _ in range(l):
    inp = list(map(float,input().split()))
    temp =[]
    for val in range(n):
        temp.append(inp[val])
    x.append(temp)
    y.append(inp[n])

out_len = int(input()) # length of samples for which the prediction is to be made
out_array=[]  
for _ in range(out_len) :
    inp = list(map(float,input().split()))
    out_array.append(inp)

# calculating the coefficient and intercept
lm = linear_model.LinearRegression()
lm.fit(x, y)
a = lm.intercept_
b = lm.coef_

# predicting the value for the given samples
for val in out_array:
    b_sum =0
    for i in range(n):
        b_sum += b[i]*val[i]
    y_pred = a + b_sum
    print(round(y_pred,2))
```







