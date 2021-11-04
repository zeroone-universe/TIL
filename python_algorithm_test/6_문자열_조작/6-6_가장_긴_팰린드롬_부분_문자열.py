def longest_palandrome(s:str):

    if len(s)<2 or s==s[::-1]:
        return s
    
    def expand(left:int, right:int)-> str:
        while left>=0 and right <len(s) and s[left]==s[right]:
            left-=1
            right+=1
        return s[left+1:right]
    
    

    result=''
    for i in range(len(s)-1):
        result=max(result,
        expand(i,i+1),
        expand(i,i+2),
        key=len)
    

    return result

input1='babadejfjefjfedksl'
print("output1 : {}".format(longest_palandrome(input1)))
input2='12345432'
print("output2 : {}".format(longest_palandrome(input2)))