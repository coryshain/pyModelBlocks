import re
from mb.util import tree


def deps2trees(buffer, format='stanford', debug=False):
    out = []
    # Regexp for extracting dependency information from a stanford dependencies file
    stan_dep = re.compile(' *[^ ]*\([^ ]+-([0-9]+) *, *([^ ]+)-([0-9]+)\)')

    # Reports whether a tree is terminal
    def term(t):
        return t.ch == []

    # Ensures that each terminal in t has a unary pre-terminal parent
    def wrap_terms(t):
        if len(t.ch) > 1:
            for i in range(len(t.ch)):
                if term(t.ch[i]):
                    t.ch[i] = tree.Tree(pos.pop(0), [tree.Tree(t.ch[i].c, [])])
                else:
                    wrap_terms(t.ch[i])
        elif len(t.ch) == 1 and len(t.ch[0].ch) == 0:
            t.c = pos.pop(0)
        return t

    # Start reading the input
    line = next(buffer)
    while line:
        # list of dependency tokens
        deps = []
        pos = []

        # Each token is on its own line, and sents are separated by newlines.
        # Reads until the end of the sentence is encountered and creates
        # a new token object for each line
        while line and not line.strip() == '':
            # Each token must have 'word', 'dep', and 'ix' fields.
            # The following lines read these in according to the
            # input format.
            if format.lower() == 'conll':
                tok = {'word': line.split()[1], 'dep': int(line.split()[7]), 'ix': int(line.split()[0])}
                pos += [str(line.split()[3])]
            elif format.lower() == 'conll-x':
                tok = {'word': line.split()[1], 'dep': int(line.split()[6]), 'ix': int(line.split()[0])}
                pos += [str(line.split()[3])]
            elif format.lower() == 'stanford':
                word = stan_dep.match(line).group(2)
                dep = stan_dep.match(line).group(1)
                ix = stan_dep.match(line).group(3)
                pos += ['X']
                tok = {'word': word, 'dep': int(dep), 'ix': int(ix)}
            else:
                raise ValueError('Unsupported format %s' % format)
            deps.append(tok)
            if debug:
                out.append('%s\n' % tok)
            line = next(buffer)

        # Dictionary of trees indexed by head sentpos
        trees = {0: tree.Tree()}

        # Add a preterminal to trees for each token in the sentence
        for tok in deps:
            trees[tok['ix']] = tree.Tree('X', [tree.Tree(tok['word'], [])])

        # Combine trees based on their dependencies (deps to 0 are the main head)
        for tok in deps:
            # Dep to 0, this is the main head
            if tok['dep'] == 0:
                trees[0] = trees[tok['ix']]
            # Dep to following head, insert tree as preceding sibling of head
            elif tok['ix'] < tok['dep']:
                trees[tok['dep']].ch.insert(-1, trees[tok['ix']])
            # Dep to preceding head, insert tree as following sibling of head
            else:
                trees[tok['dep']].ch.append(trees[tok['ix']])

        # Make sure all terminals have unary pre-terminal parents
        trees[0] = wrap_terms(trees[0])

        # Print the main tree
        out.append('%s\n' % trees[0])

        # Start reading the next sentence
        try:
            line = next(buffer)
        except StopIteration:
            line = None

    return out
