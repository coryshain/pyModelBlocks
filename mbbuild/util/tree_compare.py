from mbbuild.util import tree

def compare_trees(fG, fH):
    output = []

    def compare( tG, tH ):
        output = ''
        if tG.c == tH.c and len( tG.words() ) == len( tH.words() ) and len( tG.ch ) == len( tH.ch ) and ( len(tH.ch) < 2 or len( tG.ch[0].words() ) == len( tH.ch[0].words() ) ):
            output += ( ' (' if len(tG.ch)>0 else ' ' ) + tG.c
            for i in range( len(tH.ch) ):
                output += compare( tG.ch[i] if i<len(tG.ch) else tree.Tree(), tH.ch[i] )
            if len(tG.ch)>0: output += ')'
        else:
            output += ' <***GOLD***> ' + str(tG) + ' <**HYPOTH**> ' + str(tH) + ' <**********>'

        return output

    for i, (lineH, lineG) in enumerate(zip(fH, fG)):

        tH = tree.Tree( )
        tH.read( lineH )

        tG = tree.Tree( )
        tG.read( lineG )

        output.append('TREE ' + str(i+1) + ':\n')
        output.append(compare( tG, tH ) + '\n')

    return output
