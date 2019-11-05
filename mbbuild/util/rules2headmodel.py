from mbbuild.util import pcfg_model


def rules2headmodel(buffer):
    out = []

    R = pcfg_model.Model('R')

    for line in buffer:
        R.read(line)

    proj = {}
    proj_norms = {}

    for P in R:
        if P[0] in proj_norms:
            proj_norms[P[0]] += R[P]
        else:
            proj_norms[P[0]] = R[P]

    for P in R:
        if not P[0] in proj:
            proj[P[0]] = {}
        for ch in P[1:]:
            if ch in proj[P[0]]:
                proj[P[0]][ch] += (R[P]/len(P[1:]))/proj_norms[P[0]]
            else:
                proj[P[0]][ch] = (R[P]/len(P[1:]))/proj_norms[P[0]]

    for p in proj:
        for ch in proj[p]:
            out.append('H ' + p + ' : ' + ch + ' = ' + str(proj[p][ch]) + '\n')

    return out

