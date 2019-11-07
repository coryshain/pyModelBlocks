def split_notok_sents(notokwords, toksents):
    target = notokwords.pop(0)
    output = []
    for s in toksents:
        out = ''
        cur = ''
        for w in s.split():
            cur += w
            assert len(cur) <= len(target), '%s expected, %s provided.' % (target, cur)
            if cur == target:
                if out == '':
                    out = cur
                else:
                    out += ' ' + cur
                cur = ''
                if len(notokwords) > 0:
                    target = notokwords.pop(0)
        output.append(out + '\n')

    return output

def toks2sents(linetoks, tokmeasures):

    return split_notok_sents(list(tokmeasures.word), linetoks)
