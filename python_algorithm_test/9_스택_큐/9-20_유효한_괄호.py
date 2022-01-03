def isValid(s:str) ->bool:
    remain=[]
    table = {']':'[', ')':'(','}':'{'}

    for char in s:
        if char not in table:
            remain.append(char)
        else:
            if remain.pop()!=table[char]:
        #elif not remain or table[char] != remain.pop()
                return False
    return True


if __name__=='__main__':
    input1='([{}])'
    print(isValid(input1))