import collections

def group_anagrams(l:list):
    anagrams=collections.defaultdict(list)

    for word in l:
        anagrams[''.join(sorted(word))].append(word)


    return anagrams.values()


input=['eat','tea','tan','ate','nat','bat']
output=group_anagrams(input)
print(output)

#word='asdf'
#print(''.join(sorted(word)))