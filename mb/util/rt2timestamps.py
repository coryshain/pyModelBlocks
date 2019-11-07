def rt2timestamps(df):
    if 'fdurGP' in df.columns:
        fdur = 'fdurGP'
    else:
        fdur = 'fdur'

    df['time'] = df.groupby(['subject', 'docid'])[fdur].shift(1).fillna(value=0)
    df.time = df.groupby(['subject', 'docid']).time.cumsum() / 1000 # Convert ms to s

    return df
