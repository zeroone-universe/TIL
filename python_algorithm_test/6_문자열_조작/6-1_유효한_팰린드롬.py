import time
import collections
import re

def isPalindrome(s):
    s_list=collections.deque()
        
    for char in s:
        if char.isalnum() is True:
            s_list.append(char.lower())
            
    while len(s_list)>1:
        if s_list.popleft() !=s_list.pop():
            return False
    return True

if __name__=='__main__':
    a=isPalindrome('A man, a plan, a canal: panama')
    print(a)
    