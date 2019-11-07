import re
import string
import pandas as pd

NS_DOC_NAMES = [
    'Boar',
    'Aqua',
    'MatchstickSeller',
    'KingOfBirds',
    'Elvis',
    'MrSticky',
    'HighSchool',
    'Roswell',
    'Tulips',
    'Tourettes'
]


def ns_text_normalizer(x):
    return x.replace(
        '``', "'"
    ).replace(
        "''", "'"
    ).replace(
        "(",
        "-LRB-"
    ).replace(
        ")",
        "-RRB-"
    ).replace(
        "peaked",
        "peeked"
    )


def ns_docid_int2name(df):
    df.docid = df.docid.map(lambda i: NS_DOC_NAMES[i-1] if i < 10 else 'Other')

    return df


def docids_by_item(lineitems, tokmeasures):
    cols = ['word', 'sentid', 'sentpos', 'docid']
    outputs = []

    docix_seq = tokmeasures.item.values

    sentid = 0
    i = 0

    for row in lineitems:
        sentpos = 1
        words = row.strip().split()
        for word in words:
            outputs.append((word, sentid, sentpos, docix_seq[i]))
            sentpos += 1
            i += 1

        sentid += 1

    outputs = pd.DataFrame(outputs, columns=cols)
    outputs = ns_docid_int2name(outputs)

    return outputs


def get_onset_times(textgrid):
    onset = re.compile('            xmin = ([^ ]*)')
    offset = re.compile('            xmax = ([^ ]*)')

    timestamp = onset

    word = re.compile('            text = \" *([^ "]*)')

    wrds = []
    with open(textgrid, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('    item [2]:'):
                break
            while line and not line.startswith('            xmin ='):
                line = f.readline()

            t = timestamp.match(line).group(1)
            while line and not line.startswith('            text ='):
                line = f.readline()
            w = word.match(line).group(1)
            w = w.lower()
            if w in ['', '<s>', '</s>', '<s']:
                exclude = True
            # elif w == 'the' and t == '291.5030594213008':
            elif w == 'the' and t == '291.4':
                exclude = True
            # elif w == 'worry' and t == '140.04999':
            elif w == 'worry' and t == '140.06966742907173':
                exclude = True
            else:
                exclude = False
            w = w.translate(str.maketrans('', '', string.punctuation))
            if w == 'shrilll':
                w = 'shrill'
            if w == 'noo':
                w = 'no'
            if w == 'yess':
                w = 'yes'
            if not exclude:
                wrds.append((w, t))
            line = f.readline()
    return wrds


def textgrid2itemmeasures(itemmeasures, textgrid_dir):
    outputs = []
    columns = list(itemmeasures.columns) + ['time']
    col_map = {}
    for i in range(len(columns)):
        col_map[columns[i]] = i
    tg = []
    for i in range(10):
        f = textgrid_dir + '/' + str(i + 1) + '.TextGrid'
        tg.append(get_onset_times(f))
    tg_ix = 0
    tg_pos = 0
    for vals in itemmeasures.itertuples(index=False, name=None):
        word = vals[col_map['word']]
        word = word.replace('-LRB-', '')
        word = word.replace('-RRB-', '')
        word = word.lower().translate(str.maketrans('', '', string.punctuation))
        tg_word = tg[tg_ix][tg_pos][0]
        while len(word) > len(tg_word):
            tg_pos += 1
            tg_word += tg[tg_ix][tg_pos][0]
        assert word == tg_word, 'Mismach in document %d: %s vs. %s' % (tg_ix, word, tg_word)
        outputs.append(vals + (tg[tg_ix][tg_pos][1],))
        tg_pos += 1
        if tg_pos >= len(tg[tg_ix]):
            tg_ix += 1
            tg_pos = 0
    assert tg_ix == len(tg) and tg_pos == 0

    outputs = pd.DataFrame(outputs, columns=columns)

    return outputs


def ns_merge(evmeasures, itemmeasures):
    no_dups = [c for c in itemmeasures.columns.values if c not in evmeasures.columns.values] + ['item', 'zone', 'word']
    itemmeasures = itemmeasures.filter(items=no_dups)

    frames = []

    for s in evmeasures['subject'].unique():
        data1_s = evmeasures.loc[evmeasures['subject'] == s]
        merged = pd.merge(data1_s, itemmeasures, how='inner', on=['item', 'zone', 'word'])
        merged['subject'] = s
        frames.append(merged)
    merged = pd.concat(frames)
    merged = merged * 1 # convert boolean to [1,0]
    merged.sort_values(['subject', 'item', 'zone'], inplace=True)
    merged['startofsentence'] = (merged.sentpos == 1).astype('int')
    merged['endofsentence'] = merged.startofsentence.shift(-1).fillna(1).astype('int')
    merged['wlen'] = merged.word.str.len()

    return merged