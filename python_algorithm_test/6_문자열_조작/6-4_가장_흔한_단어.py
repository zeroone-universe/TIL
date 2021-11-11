import collections
import re

def mostcommon(paragraph, banned):
    a=[]
    for word in re.sub(r'[^\w]',' ',paragraph).lower().split():
        if word not in banned:
            a.append(word)
    
    
    b=collections.Counter(a)
    return b.most_common(1)[0][0]

if __name__=='__main__':
    paragraph='Bob hit a ball, the hit BALL flew far after it was hit'
    banned= ['hit']
    print(mostcommon(paragraph, banned))
