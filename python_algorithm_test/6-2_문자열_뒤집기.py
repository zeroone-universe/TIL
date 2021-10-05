import time

def swap1(l: list):
    left=0
    right=len(l)-1

    while(left<right):
        l[left],l[right]=l[right],l[left]
        left+=1
        right-=1

def swap2(l: list):
    l.reverse()

a=['H','a','n','n','a','h']
start = time.time()
swap2(a)
print("time :", time.time() - start)
print(a)