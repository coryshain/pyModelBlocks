import sys
import pandas as pd

def sents2sentids(lineitems):
    columns = ['word', 'sentid', 'sentpos']
    outputs = []

    sentid=0
    for line in lineitems:
        sentpos=1
        line = line.split()
        for word in line:
            outputs.append((word, sentid, sentpos))
            sentpos += 1
        sentid += 1

    outputs = pd.DataFrame(outputs, columns=columns)

    return outputs