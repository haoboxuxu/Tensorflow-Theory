import re
import collections

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(feature):
    model = collections.defaultdict(lambda: 1)
    for f in feature:
        model[f] += 1
    return model

NWORDS = train(words(open('big.txt').read()))
# print(NWORDS)

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    n = len(word)
    return set([word[0:i]+word[i+1:] for i in range(n)] +
               [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)] +
               [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] +
               [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet])
# print(edits1('leaan'))
def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])

print(correct('leaan'))
