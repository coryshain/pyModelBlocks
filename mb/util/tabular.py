import sys
import numpy as np
import pandas as pd
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser


def bin(x):
    if x <= 0:
        return 0
    if x <= 1:
        return 1
    if x <= 2:
        return 2
    if x <= 4:
        return 3
    if x <= 8:
        return 4
    return 5


def bin_dlt(df):
    for x in df.columns:
        if x.startswith('dlt'):
            df[x + 'bin'] = df[x].map(bin)

    return df


def get_surp(df):
    for x in df.columns:
        if 'prob' in x and not 'surp' in x:
            df[x + 'surp'] = -df[x]

    return df


def get_wlen(df):
    df['wlen'] = df.word.map(len)

    return df


def get_prim(df):
    for x in df.columns:
        if 'Ad' in x or 'Bd' in x:
            df[x + 'prim'] = df.map(lambda x: x[0])

    return df


def get_prevwasfix(df):
    if 'wdelta' in df.columns:
        df['prevwasfix'] = (df.wdelta == 1).as_type(int)

    return df


def augment_cols(df):
    df = bin_dlt(df)
    df = get_surp(df)
    df = get_wlen(df)
    df = get_prim(df)
    df = get_prevwasfix(df)

    return df


def merge_tables(a, b, key_cols, merge_how='inner'):
    no_dups = [c for c in b.columns.values if c not in a.columns.values]

    data2_cols = key_cols + no_dups

    merged = pd.merge(a, b.filter(items=data2_cols), how=merge_how, on=key_cols, suffixes=('', '_2'))
    merged = merged * 1 # convert boolean to [1,0]
    if 'subject' in merged.columns:
        merged.sort_values(['subject'] + key_cols, inplace=True)

    return merged


def roll(r1, r2, skip, key):
    r2.append('1')
    r1[key] = r1[key] + r2[key]
    for i in range(1, len(r1)):
        if i not in skip:
            old = r1[i]
            new = r2[i]
            try:
                old_float = float(r1[i])
                new_float = float(r2[i])
                if np.isfinite(old_float) and np.isfinite(new_float):
                    r1[i] = old_float + new_float
                elif np.isfinite(old_float):
                    r1[i] = old_float
                elif np.isfinite(new_float):
                    r1[i] = new_float
                else:
                    r1[i] = old_float + new_float
            except ValueError:
                if old in ['None', 'null']:
                    r1[i] = r2[i]
    return r1


def roll_toks(tokmeasures, itemmeasures, skip_cols=None):
    outputs = []

    if skip_cols is None:
        skip_cols = []

    gold = list(itemmeasures.itertuples(index=False, name=None))
    g = 0
    header = list(tokmeasures.columns)
    output_columns = header + ['rolled']
    skey = header.index('word')
    gkey = list(itemmeasures.columns).index('word')
    skip = [skey] + [header.index(col) for col in skip_cols if col in header]
    row = None
    for row_next in list(tokmeasures.itertuples(index=False, name=None)):
        row_next = list(row_next)
        if row is None:
            row = row_next + ['0']
        else:
            row = roll(row, row_next, skip, skey)
        assert len(row[skey]) <= len(gold[g][gkey]), 'Roll failure : %s expected, %s provided.' % (
        gold[g][gkey], row[skey])
        if row[skey] == gold[g][gkey]:
            outputs.append(row)
            row = None
            g += 1

    outputs = pd.DataFrame(outputs, columns=output_columns)

    return outputs


def rt2timestamps(df):
    if 'fdurGP' in df.columns:
        fdur = 'fdurGP'
    else:
        fdur = 'fdur'

    df['time'] = df.groupby(['subject', 'docid'])[fdur].shift(1).fillna(value=0)
    df.time = df.groupby(['subject', 'docid']).time.cumsum() / 1000 # Convert ms to s

    return df


def compute_filter(y, field, cond):
    """
    Compute filter given a field and condition

    :prm y: ``pandas`` ``DataFrame``; response data.
    :prm field: ``str``; name of column on whose values to filter.
    :prm cond: ``str``; string representation of condition to use for filtering.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    cond = cond.strip()
    if cond.startswith('<='):
        return y[field] <= (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
    if cond.startswith('>='):
        return y[field] >= (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
    if cond.startswith('<'):
        return y[field] < (np.inf if cond[1:].strip() == 'inf' else float(cond[1:].strip()))
    if cond.startswith('>'):
        return y[field] > (np.inf if cond[1:].strip() == 'inf' else float(cond[1:].strip()))
    if cond.startswith('=='):
        try:
            return y[field] == (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
        except:
            return y[field].astype('str') == cond[2:].strip()
    if cond.startswith('!='):
        try:
            return y[field] != (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
        except:
            return y[field].astype('str') != cond[2:].strip()
    raise ValueError('Unsupported comparator in filter "%s"' %cond)


def compute_filters(y, censorship_params=None):
    """
    Compute filters given a filter map.

    :prm y: ``pandas`` ``DataFrame``; response data.
    :prm censorship_params: ``dict``; maps column names to filtering criteria for their values.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    if censorship_params is None:
        return y
    select = np.ones(len(y), dtype=bool)
    for field in censorship_params:
        if field in y.columns:
            for cond in censorship_params[field]:
                select &= compute_filter(y, field, cond)
    return select


def censor(df, config_path=None):
    censorship_params = None

    if config_path is not None:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_path)

        censor = config['censor']
        censorship_params = {}
        for c in censor:
            censorship_params[c] = [x.strip() for x in censor[c].strip().split(',')]
        if len(censorship_params) == 0:
            censorship_params = None

    if censorship_params is not None:
        select = compute_filters(df, censorship_params)
        df = df[select]

    return df


def compute_splitID(y, split_fields):
    """
    Map tuples in columns designated by **split_fields** into integer ID to use for data partitioning.

    :prm y: ``pandas`` ``DataFrame``; response data.
    :prm split_fields: ``list`` of ``str``; column names to use for computing split ID.
    :return: ``numpy`` vector; integer vector of split ID's.
    """

    splitID = np.zeros(len(y), dtype='int32')
    for col in split_fields:
        splitID += y[col].cat.codes
    return splitID


def compute_partition(y, modulus, n):
    """
    Given a ``splitID`` column, use modular arithmetic to partition data into **n** subparts.

    :prm y: ``pandas`` ``DataFrame``; response data.
    :prm modulus: ``int``; modulus to use for splitting, must be at least as large as **n**.
    :prm n: ``int``; number of subparts in the partition.
    :return: ``list`` of ``numpy`` vectors; one boolean vector per subpart of the partition, selecting only those elements of **y** that belong.
    """

    partition = [((y.splitID) % modulus) <= (modulus - n)]
    for i in range(n-1, 0, -1):
        partition.append(((y.splitID) % modulus) == (modulus - i))
    return partition


def partition(df, config_path, partition):
    if not isinstance(partition, list):
        partition = [partition]

    if config_path is not None:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_path)

        params = config['partition']

        mod = params.getint('mod', 4)
        arity = params.getint('arity', 3)
        fields = params.get('fields', 'subject sentid')
        fields = fields.strip().split()

    else:
        mod = 4
        arity = 3
        fields = ['subject', 'sentid']

    for f in fields:
        df[f] = df[f].astype('category')
    cols = df.columns
    df['splitID'] = compute_splitID(df, fields)

    select = compute_partition(df, mod, arity)

    if arity == 3:
        names = ['fit', 'expl', 'held']
    elif arity == 2:
        names = ['fit', 'held']
    else:
        names = [str(x) for x in range(1, arity + 1)]

    select_new = None
    for name in partition:
        try:
            i = int(name) - 1
        except:
            i = names.index(name)
        if select_new is None:
            select_new = select[i]
        else:
            select_new |= select[i]

    return df[select_new]
