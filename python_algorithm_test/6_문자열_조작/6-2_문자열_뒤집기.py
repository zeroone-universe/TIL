def reverseString(s: list) -> None:
        left=0
        right=len(s)-1
        while left<right:
            s[left], s[right]= s[right], s[left]
            left+=1
            right-=1

if __name__=='__main__':
    a=['H','a','n','n','a','h']
    
    reverseString(a)
    print(a)