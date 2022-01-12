import itertools






def clean_seqNums(seqNums = []):
    cleaned_TX = []
    for x in seqNums:
        toRemove = []
        last = -1
        for n in x:
            if (n == last+1) or (n == last-1):
                toRemove.append(n)
            last = n
        x = [y for y in x if y not in toRemove]
        cleaned_TX.append(x)
    return cleaned_TX


def getMissedPackages(numsTX = [], numsRX = []):
    nMissed = 0
    numsTX = clean_seqNums(numsTX)
    for tx in itertools.zip_longest(numsTX,fillvalue=-1):
        found = [False for x in tx]
        for i,a in enumerate(tx):
            for r in numsRX:
                if a in r or a == -1:
                    found[i] = True
                    continue
        if not all(found):
            nMissed +=1
    return nMissed
