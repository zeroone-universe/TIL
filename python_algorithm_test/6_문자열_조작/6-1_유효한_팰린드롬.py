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
    
def palindrome2(s: str) -> bool:
    strs=collections.deque()
    for char in s:
        if char.isalnum():
            strs.append(char.lower())


    while len(strs)>1:
        if strs.popleft()!=strs.pop():
            return False

    return True
    
def palindrome3(s):
    s=s.lower()
    s=re.sub('[^a-z0-9]','',s)

    return s==s[::-1]
if __name__=='__main__':
    start = time.time()
    a=palindrome2('A man, a plan, a canal: panama')
    print(a)
    print("time :", time.time() - start)
