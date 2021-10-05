import time
import collections
import re

def palindrome1(s):
    strs=[]
    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    
    while len(strs)>1:
        if strs.pop(0)!=strs.pop():
            return False

    return True
    
def palindrome2(s):
    strs: Deque= collections.deque()
    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    
    while len(strs)>1:
        if strs.popleft()!=strs.pop(): #popleft->O(1), pop(0)->O(n) 
            return False

    return True
    
def palindrome3(s):
    s=s.lower()
    s=re.sub('[^a-z0-9]','',s)

    return s==s[::-1]

start = time.time()
a=palindrome1('A man, a plan, a canal: panama')
print(a)
print("time :", time.time() - start)
