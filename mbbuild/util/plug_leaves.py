from mbbuild.util import tree

def plug_words(t, words):
    for x in t.ch:
        if len(x.ch) == 0:
            x.c = words.pop(0)
        else:
            plug_words(x, words)

def plug_leaves(trees, words):
    t = tree.Tree()

    output = []

    for tr, wr in zip(trees, words):
        t.read(tr)
        plug_words(t, wr.split())
        output.append('%s\n' % t)

    return output
