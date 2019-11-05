from mbbuild.util import tree
from mbbuild.util import pcfg_model


def trees2deps(trees_buffer, model_buffer, debug=False):
    out = []

    heads = {}
    deps = {}

    def preterms(t):
        if preterm(t):
            return [t]
        x = []
        for ch in t.ch:
            x += preterms(ch)
        return x

    def preterm(t):
        return len(t.ch) == 1 and t.ch[0].ch == []

    def get_deps(t, ix, words):
        if preterm(t):
            deps[t] = ix
            return(words.index(t) + 1)
        heads[t] = max(t.ch, key=lambda x: head_model[t.c][x])
        if debug:
            heads[t].c = 'HEAD:' + heads[t].c + '->' + str(ix)
        children = t.ch[:]
        head = children.pop(children.index(heads[t]))
        headix = get_deps(head, ix, words)
        for ch in children:
            get_deps(ch, headix, words)
            if debug:
                ch.c += '->' + str(headix)
        return(headix)

    head_model = pcfg_model.CondModel('R')
    for line in model_buffer:
        head_model.read(line)

    t = tree.Tree()

    for line in trees_buffer:
        heads = {}
        deps = {}
        t.read(line)
        preterminals = preterms(t)
        get_deps(t, 0, preterminals)
        preterminals.insert(0, tree.Tree('X', [tree.Tree('ROOT', [])]))
        if debug:
            out.append(str(t) + '\n')
        for i in range(1, len(preterminals)):
            out.append('X(' + preterminals[deps[preterminals[i]]].ch[0].c + '-' + str(deps[preterminals[i]]) + ', ' + str(preterminals[i].ch[0].c) + '-' + str(i) + ')\n')
        out.append('\n')

    return out

 
