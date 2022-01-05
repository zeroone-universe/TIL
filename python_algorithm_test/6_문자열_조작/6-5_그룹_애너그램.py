import collections

def group_anagrams(words: list):
    anagrams=collections.defaultdict(list)

    for word in words:
        anagrams[''.join(sorted(word))].append(word)
    

    return list(anagrams.values())

if __name__=='__main__':
    inp=['eat','tea','tan','ate','nat','bat']

    print(group_anagrams(inp))