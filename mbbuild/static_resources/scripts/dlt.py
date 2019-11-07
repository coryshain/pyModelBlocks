import re, os, sys, argparse
from mbbuild.util import tree

argparser = argparse.ArgumentParser(description='''
Calculates Dependency Locality Theory integration cost values by word (along with some other useful fields).

Integration cost is implemented in 8 ways by varying both of the following binary parameters:

+/- CoordMod (c): Preceding conjuncts are skipped and only the weight of the heaviest conjunct is considered when calculating intervening discourse referents.
+/- VerbMod (v): Finite verbs weigh 2 EU (instead of 1) and non-finite verbs weigh 1 EU (instead of zero).
+/- ModMod (m): Preceding modifiers do not count as dependencies (and their distances are ommitted from the total integration cost).

Corresponding column names are "dlt(c)(v)(m)", with the letter present if the corresponding mod value is "+", and absent if the corresponding mod value is "-". For example, column "dltcm" is the integration cost using parameters +CoordMod, -VerbMod, +ModMod.

Input: Generalized Categorial Grammar (GCG15 or later) linetrees through stdin.
Output: A space-delimited data table with a row for each word in the input.
''')
argparser.add_argument('-d', '--debug', dest='DEBUG', action='store_true', help='Include debugging information about the calculation of dependencies in the output')
args, unknown = argparser.parse_known_args()

punc = ["-LRB-", "-RRB-", ",", "-", ";", ":", "\'", "\'\'", '\"', "`", "``", ".", "!", "?", "*FOOT*"]
phrasePunc = ["-RRB-", ",", "-", ";", ":", ".", "!", "?"]
pronouns = ['i', 'he', 'she', 's/he', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them', \
        'who', 'whom', 'which', 'that', 'myself', 'himself', 'herself', 'itself', 'ourselves', 'themselves', 'oneself']
objExt = re.compile('.*-a.*-g.*')
terms = []
DLTcosts = []
DLTcostsV = []
coords = []
ends = []
complete = []
cCosts = []
cvCosts = []
post = []
postpost = []

def terminal(T):
    return len(T.ch) == 1 and len(T.ch[0].ch) == 0

def isPunc(t):
    if len(t.ch) == 0:
        if t.c not in punc:
            return False
        return True
    punctuation = True
    for c in t.ch:
        punctuation = punctuation & isPunc(c)
    return punctuation

def isPhrasePunc(t):
    if len(t.ch) == 0:
        if t.c not in phrasePunc:
            return False
        return True
    punctuation = True
    for c in t.ch:
        punctuation = punctuation & isPhrasePunc(c)
    return punctuation


def argsFromString(string):
    if string[-1] != '-':
        string += '-' #the hyphen triggers a push to the list, so this ensures last arg is pushed
    argMap = {}
    currentOp = 'head'
    if string[0:5] == '-LRB-' or string[0:5] == '-RRB-':
        argMap[currentOp] = [string[0:5]]
        string = string[5:]
        currentOp = ''
    currentArg = ''
    braceDepth = 0
    
    for char in string:
        if char == '-' and braceDepth == 0:
            if currentArg != '':
                if currentOp in argMap:
                    argMap[currentOp].append(stripBraces(currentArg))
                else:
                    argMap[currentOp] = [stripBraces(currentArg)]
            currentOp = ''
            currentArg = ''
        else:
            if currentOp == '':
                currentOp = char
            else:
                if char == '{':
                    braceDepth += 1
                if char == '}':
                    braceDepth -= 1
                currentArg += char
    return argMap

def stripBraces(string):
    if string[0] == '{':
        return string[1:-1]
    else:
        return string
        
def getPOS(T):
    if T.c[0] in punc:
        return 'null'
    if T.ch[0].c in pronouns:
        return 'pronoun' 
    if T.c == 'N-b{N-aD}':
        return 'D'
    if T.c[0] in ['B', 'G', 'L']:
        return 'Vnon'
    if T.c[0] in ['N', 'V', 'A', 'R', 'D', 'X']:
        return T.c[0]
    return 'O'
        
def head(T):
    x = T
    while not terminal(x):
        if (len(x.ch)) == 1:
            x = x.ch[0]
        else:
            lArgs = argsFromString(x.ch[0].c)
            rArgs = argsFromString(x.ch[1].c)
            if 'l' not in lArgs:
                if x.ch[0].c == 'N-b{N-aD}':
                    x = x.ch[1]
                else:
                    x = x.ch[0]
            elif 'l' not in rArgs:
                if lArgs['l'] == ['M'] and 'p' in rArgs and rArgs['p'] == ['Pc']:
                    x = x.ch[0]
                else:
                    x = x.ch[1]
            else:
                raise ValueError('-l rule specified on both children: ' + str(x.p))
    return x
    
def maxProj(T):
    x = T
    selfArgs = argsFromString(x.c)
    if x.p is not None and 'l' not in selfArgs and x.c != 'N-b{N-aD}' and (not x.c.startswith('X-cX-dX') or x.p.c.startswith('X-cX-dX')):
        return maxProj(x.p)
    else:
        return x

def first(T):
    x = T
    while not terminal(x):
        x = x.ch[0]
    return x

def last(T):
    x = T
    while not terminal(x):
        x = x.ch[-1]
    return x

def ancestor(T1, T2):
    x = T1
    while x.p != None:
        if x.p == T2:
            return True
        x = x.p
    return False

def objGap(t):
    if len(t.p.ch) < 2 or t != t.p.ch[1]:
        return False
    label = argsFromString(t.c)
    slabel = argsFromString(t.p.ch[0].c)
    if 'l' in slabel and 'N' in slabel['l'] \
            and 'a' in label and 'N' in label['a'] \
            and 'g' in label and 'N' in label['g']:
        return True


def calcCosts(t, MOD):
    tmp = t
    while t.p != None and len(t.p.ch) == 1:
        t = t.p
    s = getIdx(t) + 1
    dlt = 0
    dltc = 0
    dltcv = 0
    dltv = 0
    stop = isPunc(t) or t.c[0] == 'X' or t.p is None
    s = None
    depdirSyn = 'null'
    depdirSem = 'null'
    incr = False
    objGap = False
    while not stop:
        label = argsFromString(t.c)
        if not objGap:
            objGap = 'g' in label and 'N' in label['g'] and 'a' in label and 'N' in label['a']
        nonhead = ('l' in label and not (t.sibling() and t.sibling().c == 'N-b{N-aD}') and not 'C' in label['l']) or t.c.startswith('N-b{N-aD}')
        coord = ('l' in label and 'C' in label['l']) or ('c' in label) 
        if len(t.p.ch) == 2 and t == t.p.ch[1] and not isPunc(t.p.ch[0]) and not coord:
            slabel = argsFromString(t.p.ch[0].c)
            if MOD:
                if 'l' in slabel and 'M' in slabel['l']:
                    t = t.p
                    stop = nonhead or t.p is None
                    continue
            if 'l' in slabel and 'N' in slabel['l']:
                t = t.p
                stop = nonhead or t.p is None
                continue
            s = head(t.p.ch[0])
            e = tmp
            if 'l' in label:
                if 'M' in label['l']:
                    depdirSem = 'bwd'
                else:
                    depdirSem = 'fwd'
            else:
                depdirSem = 'bwd'
            if nonhead:
                if 'N' in label['l'] and objGap:
                    if args.DEBUG:
                        print('=====')
                        print('Object gap from "%s" to "%s"' % (e.ch[0].c, s.ch[0].c))
                        print('=====')
                    incr = True
                    depdirSyn = 'bwd'
                else:
                    depdirSyn = 'fwd'
            else:
                depdirSyn = 'bwd'
            if args.DEBUG:
                if incr:
                    wordlist = ' '.join([term.ch[0].c for term in terms[getIdx(s)+1:getIdx(e) + 1]])
                else:
                    wordlist = ' '.join([term.ch[0].c for term in terms[getIdx(s)+1:getIdx(e)]])
                if wordlist == '':
                    wordlist = '<EMPTY>'
                print('=====')
                print('Dependency with word "%s" in preceding subtree: %s' % (s.ch[0].c, t.p.ch[0]))
                print('Current word: ' + e.ch[0].c)
                print('List of intervening words: ' + wordlist)
                print('Dependency direction: ' + str(depdirSyn))
                print('=====')
            dlt += sumCosts(s, e, False, False, incr)
            dltc += sumCosts(s, e, False, True, incr)
            dltcv += sumCosts(s, e, True, True, incr)
            dltv += sumCosts(s, e, True, False, incr)
        t = t.p
        stop = nonhead or t.p is None
    return dlt, dltc, dltcv, dltv, depdirSyn, depdirSem
        
def getNext(t):
    tmp = t
    next = None
    while t.p != None and t == t.p.ch[-1]:
        t = t.p
    while not terminal(t):
        t = t.ch[-1]
    if t != tmp and terminal(t):
        next = t
    return next
    
def sumCosts(T1, T2, vMod, cMod, incr):
    s = getIdx(T1) + 1
    e = getIdx(T2)
    if incr:
        e += 1
    if cMod:
        if len(coords) > 0:
            for c in coords:
                if not ancestor(T1, c) and ancestor(T2, c):
                    return sumBetween(s, getIdx(first(c)), vMod, cMod) + sumBetween(getIdx(first(maxProj(T2))), e, vMod, cMod)
    if objExt.match(T2.c):
        e = getIdx(T2) + 1
    return sumBetween(s, e, vMod, cMod)
    
def sumBetween(s, e, vMod, cMod):
#    if cMod:
#        print DLTcosts[s:e]
#        print ' '.join([t.ch[0].c for t in terms[s:e]])
    sum = 0
    i = s
    if cMod:
        for j in range(0, len(complete)):
            cs = getIdx(first(complete[j]))
            ce = getIdx(last(complete[j]))
            if e < ce:
                break
            while i < cs:
                if vMod:
                    sum += DLTcostsV[i]
                else:
                    sum += DLTcosts[i]
                i += 1
            if i <= ce:
                i = ce + 1
                if vMod:
                    sum += cvCosts[j]
                else:
                    sum += cCosts[j]
        while i < e:
            if vMod:
                sum += DLTcostsV[i]
            else:
                sum += DLTcosts[i]
            i += 1
    else:
        if vMod:
            for i in range(s, e):
                sum += DLTcostsV[i]
        else:
            for i in range(s, e):
                sum += DLTcosts[i]
    return sum
    
def getIdx(T):
    while not terminal(T) and len(T.ch) == 1:
        T = T.ch[0]
    if not terminal(T):
        raise ValueError('Cannot get index of non-terminal tree')
    return terms.index(T)    
    
def cWeight(T, vMod):
    isCoord = False
    if terminal(T):
        if T.c[0] == 'N' and T.c != 'N-b{N-aD}':
            return 1
        if vMod:
            if T.c[0] == 'V':
                return 2
            if T.c[0] == 'G' or T.c[0] == 'L' or T.c[0] == 'B':
                return 1
        else:
            if T.c[0] == 'V':
                return 1
        return 0
    if len(T.ch) == 2:
        for t in T.ch:
            if "-c" in t.c:
                isCoord = True
    if isCoord:
        return max(cWeight(T.ch[0], vMod), cWeight(T.ch[1], vMod))
    weight = 0
    for t in T.ch:
        weight += cWeight(t, vMod)
    return weight
    
def printToks(T):
    global post, postpost
    if terminal(T):
        terms.append(T)
        discCost = 0
        discCostV = 0
        if T.c[0] == 'N' and T.c[0:9] != 'N-b{N-aD}' and T.ch[0].c.lower() not in pronouns:
            discCost = 1
            discCostV = 1
        elif T.c[0] == 'V' or T.c[0] == 'Q':
            discCost = 1
            discCostV = 2
        elif T.c[0] == 'G' or T.c[0] == 'L' or T.c[0] == 'B':
            discCostV = 1
        DLTcosts.append(discCost)
        DLTcostsV.append(discCostV)
       
        if args.DEBUG:
            print('=====')
            print('-ModMod Dependencies')
            print('=====')
        ic, icc, iccv, icv, depdirSyn, depdirSem = calcCosts(T, False)
        dlt, dltc, dltcv, dltv = [discCost + ic, discCost + icc, discCostV + iccv, discCostV + icv]
        if args.DEBUG:
            print('=====')
            print('+ModMod Dependencies')
            print('=====')
        icm, iccm, iccvm, icvm, depdirSynM, depdirSemM = calcCosts(T, True)
        dltm, dltcm, dltcvm, dltvm = [discCost + icm, discCost + iccm, discCostV + iccvm, discCostV + icvm]
        if len(coords) > 0:
            if T == ends[-1]:
                while len(complete) > 0:
                    if ancestor(complete[-1], coords[-1]):
                        complete.pop()
                        cCosts.pop()
                        cvCosts.pop()
                    else:
                        break
                complete.append(coords[-1])
                cCosts.append(cWeight(complete[-1], False))
                cvCosts.append(cWeight(complete[-1], True))
                coords.pop()
                ends.pop()
        print(T.ch[0].c + ' ' + str(discCost) + ' ' + str(discCostV) + ' ' \
              + str(dlt) + ' ' + str(dltc) + ' ' + str(dltcv) + ' ' + str(dltv) + ' ' \
	      + str(dltm) + ' ' + str(dltcm) + ' ' + str(dltcvm) + ' ' + str(dltvm) + ' ' \
              + str(getPOS(T)) + ' ' + str(depdirSyn) + ' ' + str(depdirSem) + ' ' + str(depdirSynM) + ' ' + str(depdirSemM) + ' ' \
              + str(int(isPhrasePunc(T.ch[0]))))
    else:
        if len(T.ch[0].c) > 1 and T.ch[0].c.endswith('lC') and not '-c' in T.c:
            coords.append(T)
            ends.append(last(T))
        for t in T.ch:
            printToks(t)

print('word dltdc dltdcv ' \
    + 'dlt dltc dltcv dltv ' \
    + 'dltm dltcm dltcvm dltvm ' \
    + 'pos depdirSyn depdirSem depdirSynM depdirSemM' + ' ' \
    + 'punc')

for line in sys.stdin:
    if (line.strip() !='') and (line.strip()[0] != '%'):
        terms = []
        DLTcosts = []
        DLTcostsV = []
        coords = []
        ends = []
        complete = []
        cCosts = []
        cvCosts = []
        post = [0, 0, 0, 0, 0, 0, 0, 0]
        postpost = [0, 0, 0, 0, 0, 0, 0, 0]
        T = tree.Tree()
        T.read(line)
        printToks(T)
