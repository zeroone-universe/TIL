import collections
import re


def mostCommonWord(paragraph: str, banned: list) -> str:
    paragraph_list=[]
    for word in re.sub(r'[^\w]', ' ', paragraph).lower().split():
        if word not in banned:
            paragraph_list.append(word)
    counts=collections.Counter(paragraph_list)
    
    return counts.most_common(1)[0][0]


if __name__=='__main__':
    paragraph='Bob hit a ball, the hit BALL flew far after it was hit'
    banned= ['hit']
    print(mostCommonWord(paragraph, banned))
