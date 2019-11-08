import sys
import os
import argparse


def nested(A, B, max_df=1):
    # Only checks nesting based on variable names and ablation configuration
    # Does not check for identity of data or formula
    a_vars = A.split('.')[-3].split('_')[2:-1]
    b_vars = B.split('.')[-3].split('_')[2:-1]
    if len(a_vars) != len(b_vars):
        return A, B, False
    a_vars = sorted([(x[1:],0) if x.startswith('~') else (x,1) for x in a_vars], key=lambda x:x[0])
    b_vars = sorted([(x[1:],0) if x.startswith('~') else (x,1) for x in b_vars], key=lambda x:x[0])
    if set([x[0] for x in a_vars]) != set([x[0] for x in b_vars]):
        return A, B, False
    baseline = None
    full = None
    df = 0
    for a, b in zip(a_vars, b_vars):
        if a[1] != b[1]:
            df += 1
            if df > max_df:
                return A, B, False
            if a[1] < b[1]:
                baseline_cur = A
                full_cur = B
            else:
                baseline_cur = B
                full_cur = A
            if baseline is None:
                baseline = baseline_cur
                full = full_cur
            elif baseline != baseline_cur:
                return A, B, False

    return baseline, full, True


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run significance tests on pairs of nested models with 1 degree of freedom.
    ''')
    argparser.add_argument('paths', help='Paths to prediction files (LME model paths are inferred from the prediction path).')
    argparser.add_argument('-t', '--type', type=str, default='lrt', help='Type of test to use. One of ["lrt", "pt"], for likelihood ratio and permutation testing, respectively.')
    args = argparser.parse_args()

    paths = sorted(args.paths, key=len, reverse=True)

    models = []

    for path in paths:
        path_chunks = path.split('.')
        args = path_chunks[-2].split('-')
        args = [x for i, x in enumerate(args) if i != 5]
        m = '.'.join(path_chunks[:-2] + ['-'.join(args), 'reg'])
        args = path.split('.')[-2].split('-')
        fit_part = args[2]
        eval_part = args[5]
        assert fit_part == eval_part, 'Likelihood ratio testing is in-sample and requires matched training and evaluation partitions. Saw "%s", "%s"' % (fit_part, eval_part)
        models.append(m)


    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            baseline, full, is_nested = nested(m1, m2)
            if is_nested:
                os.system('mb/static_resources/scripts/signif-%s.R %s %s' % (m1, m2, args.type))

