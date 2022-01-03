def removeDuplicateLetters(s:str)->str:
    table=[]
    for char in s:
        if char not in table:
            table.append(char)
    
    out=''.join(sorted(table))
    return out

if __name__=='__main__':
    output1=removeDuplicateLetters('bcabc')
    output2=removeDuplicateLetters('bcazhjqkjbc')
    print(output1, output2)