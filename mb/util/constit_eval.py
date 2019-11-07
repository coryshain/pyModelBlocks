import re, math
from mb.util import tree

# Extension of tree.Tree() to provide useful character-level functionality
# that permits evaluation and comparison of trees even when their word
# tokenizations differ.
class CharTree(tree.Tree, object):
  def __init__(self, c='', ch=[], p=None,l=0,r=0,lc=0,rc=0):
    super(CharTree, self).__init__(c, ch, p, l, r)
    self.lc = lc
    self.rc = rc

  # obtain tree from string
  def read(self,s,fIndex=0,cIndex=0):
    self.ch = []
    # parse a string delimited by whitespace as a terminal branch (a leaf)
    m = re.search('^ *([^ ()]+) *(.*)',s)
    if m != None:
        (self.c,s) = m.groups()
        self.l = fIndex
        self.lc = cIndex
        self.r = fIndex
        self.rc = self.lc + len(self.c) - 1
        return s, fIndex+1, cIndex + len(self.c)
    # parse nested parens as a non-terminal branch
    m = re.search('^ *\( *([^ ()]*) *(.*)',s)
    if m != None:
        (self.c,s) = m.groups()
        self.l = fIndex
        self.lc = cIndex
        # read children until close paren
        while True:
            m = re.search('^ *\) *(.*)',s)
            if m != None:
                return m.group(1), fIndex, cIndex
            t = CharTree()
            s, fIndex, cIndex = t.read(s, fIndex, cIndex)
            self.ch += [t]
            t.p = self
            self.r = t.r
            self.rc = t.rc
    return ''

# Get dicts of spans, categories, and depths from a given tree
#
#  Params:
#    t: a tree representation (Char_Tree() only, will not work with tree.Tree())
#    spans: a dictionary mapping a 2-tuple of ints (<first_index, last_index>)
#           representing a character span to fields 'wlen' (length in words of 
#           span), 'cat' (category of span), and 'wspan' (a 2-tuple of ints 
#           representing the same span in words)
#    cats: a dictionary mapping categories to a list of the char spans that they label
#    depths: a dictionary mapping char spans to left-corner depths to counts
#    depth: the left-corner depth of the current tree
#    right_parent: the current tree's parent was a right child (boolean)
#
#  Return:
#    spans, cats, and depths (updates of params)
#
def process_tree(t, spans=None, cats=None, depths=None, depth=1, right_parent=False, minlength=2):
  if spans is None:
    spans = {}
  if cats is None:
    cats = {}
  if depths is None:
    depths = {}
  if (t.lc,t.rc) not in spans:
    spans[(t.lc,t.rc)] = {'wlen': t.r-t.l+1, 'cat': t.c, 'wspan':(t.l,t.r)}
  if (t.lc,t.rc) not in depths and t.r-t.l+1 >= minlength:
    depths[(t.lc,t.rc)] = depth 
  if t.c not in cats:
    cats[t.c] = [(t.lc,t.rc)]
  else:
    cats[t.c] += [(t.lc,t.rc)]
  for i in range(len(t.ch)):
    if right_parent and i == 0 and len(t.ch) > 1:
      newdepth = depth + 1
    else:
      newdepth = depth
    process_tree(t.ch[i], spans, cats, depths, newdepth, i>0, minlength=minlength)
  return spans, cats, depths
     
  
# Get a sequential list of the preterminal nodes in a tree
#
#  Params:
#    t: a tree representation (tree.Tree() or Char_Tree())
#
#  Return:
#    a list of the preterminal nodes in t
def preterms(t):
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    return [t.c]
  else:
    tmp = []
    for ch in t.ch:
      tmp += preterms(ch)
    return tmp

# Get an integer representation of a tag sequence.
# Maps each unique tag type in tags to a unique
# integer label in the output.
#
#  Params:
#    tags: a sequential list of tags
#
#  Return:
#    ints: an integer representation of tags
def tags2ints(tags):
  legend = {}
  ints = []
  i = 1
  for a in tags:
    if a in legend:
      ints.append(legend[a])
    else:
      ints.append(i)
      legend[a] = i
      i += 1
  return ints

# Get a binary segmentation list from a word list.
# Maps each non-space character to 1 (first character in
# in a segment) or 0 (otherwise).
#
#  Params:
#    wrds: a list of the words/tokens in a sequence
#
#  Return:
#    A binary segmentation list representation of wrds
def get_seg(wrds):
  out = []
  chars = ' '.join(wrds)
  seg = True
  for i in range(len(chars)):
    if (chars[i]) == ' ':
      seg = True
    else:
      out.append(int(seg))
      seg = False
  return out

# ACCURACY EVAL FUNCTIONS:

# Get precision, recall, and F1
#
#  Params:
#    tp: count of true positives
#    fp: count of false positives
#    fn: count of false negatives
#
#  Return:
#    p: precision
#    r: recall
#    f1: f-measure
def accuracy(tp, fp, fn):
  if tp + fp + fn == 0:
    return 1, 1, 1
  elif tp == 0:
    return 0, 0, 0
  else:
    p = float(tp) / float(tp + fp)
    r = float(tp) / float(tp + fn)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1

# Get marginal entropy of a set of classes
#
#  Params:
#    A: dictionary from classes (C) to clusters (K) to counts
#    n: total number of data points
#
#  Return:
#    H(C) (a real number)
def H_C(A, n):
  H = float(0)
  for c in A:
    exp_c = float(sum([A[c][k] for k in A[c]])) / n
    H += exp_c * math.log(exp_c)
  return -H

# Get conditional entropy of a set of classes given
# a set of clusters
#
#  Params:
#    A: a dictionary from clusters (K) to classes (C) to counts
#    n: total number of data points
#
#  Return:
#    H(C|K) (a real number)
def H_CgivK(A, n):
  H = float(0)
  for k in A:
    denom = float(sum([A[k][c] for c in A[k]]))
    for c in A[k]:
      H += float(A[k][c])/n * math.log(float(A[k][c])/denom)
  return -H

# Get homogeneity score h (Rosenberg & Hirschberg 2007)
#
#  Params:
#    A: a dictionary from clusters (K) to classes (C)
#    Aprime: a dictionary from classes (C) to clusters (K)
#    n: total number of data points
def homog(A, Aprime, n):
  hc = H_C(Aprime, n)
  if hc == 0:
    return 1
  else:
    return 1 - (H_CgivK(A, n) / hc)

# Get many-to-one accuracy (raw accuracy given a mapping
# of clusters to most frequent classes)
#
#  Params:
#    A: a dictionary from classes (C) to clusters (K)
#
#  Return:
#    many-to-one accuracy (a real number)
def m21(A):
  correct = 0
  total = 0
  for c in A:
    max = 0
    argMax = None
    for k in A[c]:
      total += A[c][k]
      if A[c][k] > max:
        max = A[c][k]
        argMax = k
    correct += max
  return float(correct) / total

# Main program
def constit_eval(gold, test, minlength=2, debug=False):
  # Initialize global variables
  output = []
  g = CharTree()
  t = CharTree()
  all_tp = all_fp = all_fn = 0
  brac_tp = brac_fp = brac_fn = 0
  seg_tp = seg_fp = seg_fn = 0
  n_wrds = 0
  max_proj = ['NP', 'VP', 'ADJP', 'ADVP', 'PP', 'S']
  max_proj_scores = {}
  for cat in max_proj:
    max_proj_scores[cat] = {}
    max_proj_scores[cat]['hit'] = 0
    max_proj_scores[cat]['miss'] = 0
  sentid = 1
  same_g2t = {}
  same_t2g = {}
  t_cat_counts = {}
  leaves_g2t = {}
  leaves_t2g = {}
  depth_counts = {}
  depth_length_counts = {}

  # Read the next line from each dataset
  g_line = next(gold)
  while g_line and g_line.strip() == '':
    g_line = next(gold)

  t_line = next(test)
  while t_line and t_line.strip() == '':
    t_line = next(test)

  # Process each pair of trees
  while g_line and t_line:
    # Read in trees and get word and character lists from them
    g.read(g_line)
    g_wrds = g.words()
    g_chrs = ''.join(g_wrds)
    t.read(t_line)
    t_wrds = t.words()
    t_chrs = ''.join(t_wrds)
    if t_line.startswith('(FAIL'):
      g_line = next(gold)
      t_line = next(test)
      continue
    assert g_chrs == t_chrs, 'Gold sentence (%s) differs from test sentence (%s)' %(' '.join(g.words()), ' '.join(t.words()))

    # Get data about character spans and their labels
    seg_g = get_seg(g_wrds)
    g_cat_by_spans, g_spans_by_cat = process_tree(g, {}, {}, {}, minlength=minlength)[:2]
    t_cat_by_spans, t_spans_by_cat, t_depth_by_spans = process_tree(t, {}, {}, {}, minlength=minlength)
    g_spans = g_cat_by_spans.keys()
    t_spans = t_cat_by_spans.keys()
    all_same = set(g_spans) & set(t_spans)

    # Get list of terminals in test and gold
    g_leaves = [span for span in g_spans if g_cat_by_spans[span]['wlen'] == 1]
    t_leaves = [span for span in t_spans if t_cat_by_spans[span]['wlen'] == 1]
    leaves_same = set(g_leaves) & set(t_leaves)
    n_wrds += len(leaves_same)

    # Get list of complex non-terminals in test and gold
    g_spans_complex = [span for span in g_spans if g_cat_by_spans[span]['wlen'] >= minlength]
    t_spans_complex = [span for span in t_spans if t_cat_by_spans[span]['wlen'] >= minlength]
    # Remove constituents over bad token segmentations (i.e. all constituents
    # whose edges fall on characters that are not segment boundaries in the gold).
    # Avoids double-punishing the bracketing for segmentation errors.
    t_spans_complex = [span for span in t_spans_complex if seg_g[span[0]] == 1 and (span[1] == len(seg_g) - 1 or seg_g[span[1]+1] == 1)]
    # Remove spans from test that were analyzed as complex but
    # correspond to leaves in the gold.
    # Avoids punishing the bracketing for oversegmentation.
    t_spans_complex = set(t_spans_complex) - set(g_leaves)
    for span in [span for span in t_spans if t_cat_by_spans[span]['wlen'] >= minlength]:
      if t_cat_by_spans[span]['cat'] in t_cat_counts:
        t_cat_counts[t_cat_by_spans[span]['cat']] += 1
      else:
        t_cat_counts[t_cat_by_spans[span]['cat']] = 1
    brac_same = set(g_spans_complex) & set(t_spans_complex)

    # Get depth counts
    depth_counts_sent = {}
    depth_length_counts_sent = {}
    for span in t_depth_by_spans:
      cur_depth = t_depth_by_spans[span]
      span_length = t_cat_by_spans[span]['wlen']
      if debug:
        if cur_depth in depth_counts_sent:
          depth_counts_sent[cur_depth] += 1
        else:
          depth_counts_sent[cur_depth] = 1
      if cur_depth in depth_counts:
        depth_counts[cur_depth] += 1
      else:
        depth_counts[cur_depth] = 1
      if debug:
        if cur_depth in depth_length_counts_sent:
          if span_length in depth_length_counts_sent[cur_depth]:
            depth_length_counts_sent[cur_depth][span_length] += [' '.join(t_wrds[t_cat_by_spans[span]['wspan'][0]:t_cat_by_spans[span]['wspan'][1]+1])]
          else:
            depth_length_counts_sent[cur_depth][span_length] = [' '.join(t_wrds[t_cat_by_spans[span]['wspan'][0]:t_cat_by_spans[span]['wspan'][1]+1])]
        else:
          depth_length_counts_sent[cur_depth] = {span_length: [' '.join(t_wrds[t_cat_by_spans[span]['wspan'][0]:t_cat_by_spans[span]['wspan'][1]+1])]}
      if cur_depth in depth_length_counts:
        if span_length in depth_length_counts[cur_depth]:
          depth_length_counts[cur_depth][span_length] += 1
        else:
          depth_length_counts[cur_depth][span_length] = 1
      else:
        depth_length_counts[cur_depth] = {span_length: 1}


    # Get maximal projection discovery scores
    if debug:
      debug_maxproj = 'Maximal projection discovery:\n'
    for cat in max_proj_scores.keys():
      if cat in g_spans_by_cat:
        for span in g_spans_by_cat[cat]:
          if g_cat_by_spans[span]['wlen'] >= minlength:
            if debug:
              debug_maxproj += '  ' + cat + ' span: ' + ' '.join(t_wrds[g_cat_by_spans[span]['wspan'][0]:g_cat_by_spans[span]['wspan'][1] + 1]) + '\n'
            if span in t_cat_by_spans:
              max_proj_scores[cat]['hit'] += 1
              if debug:
                debug_maxproj += '    Found: YES\n'
            else:
              max_proj_scores[cat]['miss'] += 1
              if debug:
                debug_maxproj += '    Found: NO\n'

    # Update cluster-label matrix (constituents)
    for span in brac_same:
      if g_cat_by_spans[span]['cat'] in same_g2t:
        if t_cat_by_spans[span]['cat'] in same_g2t[g_cat_by_spans[span]['cat']]:
          same_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] += 1
        else:
          same_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] = 1
      else:
        same_g2t[g_cat_by_spans[span]['cat']] = {t_cat_by_spans[span]['cat']: 1}
      if t_cat_by_spans[span]['cat'] in same_t2g:
        if g_cat_by_spans[span]['cat'] in same_t2g[t_cat_by_spans[span]['cat']]:
          same_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] += 1
        else:
          same_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] = 1
      else:
        same_t2g[t_cat_by_spans[span]['cat']] = {g_cat_by_spans[span]['cat']: 1}

    # Update cluster-tag matrix (parts-of-speech)
    for span in leaves_same:
      if g_cat_by_spans[span]['cat'] in leaves_g2t:
        if t_cat_by_spans[span]['cat'] in leaves_g2t[g_cat_by_spans[span]['cat']]:
          leaves_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] += 1
        else:
          leaves_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] = 1
      else:
        leaves_g2t[g_cat_by_spans[span]['cat']] = {t_cat_by_spans[span]['cat']: 1}
      if t_cat_by_spans[span]['cat'] in leaves_t2g:
        if g_cat_by_spans[span]['cat'] in leaves_t2g[t_cat_by_spans[span]['cat']]:
          leaves_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] += 1
        else:
          leaves_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] = 1
      else:
        leaves_t2g[t_cat_by_spans[span]['cat']] = {g_cat_by_spans[span]['cat']: 1}

    # Update scores for overall (constituent span) accuracy
    all_tp += len(all_same)
    all_t_only = len(t_spans) - len(all_same)
    all_fp += all_t_only
    all_g_only = len(g_spans) - len(all_same)
    all_fn += all_g_only

    # Update scores for segmentation
    seg_tp += len(leaves_same)
    seg_t_only = len(t_leaves) - len(leaves_same)
    seg_fp += seg_t_only
    seg_g_only = len(g_leaves) - len(leaves_same)
    seg_fn += seg_g_only

    # Update scores for bracketing accuracy
    brac_tp += len(brac_same)
    brac_t_only = len(t_spans_complex) - len(brac_same)
    brac_fp += brac_t_only
    brac_g_only = len(g_spans_complex) - len(brac_same)
    brac_fn += brac_g_only

    # Print debugging/verbose log for this sentence
    if debug:
      all_p, all_r, all_f1 = accuracy(len(all_same), all_t_only, all_g_only)
      seg_p, seg_r, seg_f1 = accuracy(len(leaves_same), seg_t_only, seg_g_only)
      brac_p, brac_r, brac_f1 = accuracy(len(brac_same), brac_t_only, brac_g_only)
      output.append('=================================\n')
      output.append('Evaluating sentence #%d:\n' %sentid)
      output.append('Unsegmented input: %s\n' %g_chrs)
      output.append('Gold segmentation: %s\n' %' '.join(g_wrds))
      output.append('Gold tree: %s\n' %str(g))
      output.append('Test segmentation: %s\n' %' '.join(t_wrds))
      output.append('Test tree: %s\n' %str(t))
      output.append('\n')
      output.append('Overall (char spans):\n')
      output.append('  Num gold character spans: %d\n' %len(g_spans))
      output.append('  Num test character spans: %d\n' %len(t_spans))
      output.append('  Matching: %d\n' %len(all_same))
      output.append('  False positives (in test but not gold): %d\n' %all_t_only)
      output.append('  False negatives (in test but not gold): %d\n' %all_g_only)
      output.append('  Precision: %.4f\n' %all_p)
      output.append('  Recall: %.4f\n' %all_r)
      output.append('  F-Measure: %.4f\n' %all_f1)
      output.append('\n')
      output.append('Segmentation:\n')
      output.append('  Num gold words: %d\n' %len(g_leaves))
      output.append('  Num test words: %d\n' %len(t_leaves))
      output.append('  Matching: %d\n' %len(leaves_same))
      output.append('  False positives (in test but not gold): %d\n' %seg_t_only)
      output.append('  False negatives (in test but not gold): %d\n' %seg_g_only)
      output.append('  Precision: %.4f\n' %seg_p)
      output.append('  Recall: %.4f\n' %seg_r)
      output.append('  F-Measure: %.4f\n' %seg_f1)
      output.append('\n')
      output.append('Bracketing:\n')
      output.append('  (Ignoring constituents of length < %d)\n' %minlength)
      output.append('  Gold constituents:\n    \n' + '\n    '.join([' '.join(g_wrds[g_cat_by_spans[x]['wspan'][0]:g_cat_by_spans[x]['wspan'][1]+1]) for x in g_spans_complex]))
      output.append('  Test constituents:\n    \n' + '\n    '.join([' '.join(t_wrds[t_cat_by_spans[x]['wspan'][0]:t_cat_by_spans[x]['wspan'][1]+1]) for x in t_spans_complex]))
      output.append('  Test constituents by depth:\n')
      for depth in depth_counts_sent:
        output.append('    Depth = %d:\n      \n' %depth + '\n      '.join([' '.join(t_wrds[t_cat_by_spans[x]['wspan'][0]:t_cat_by_spans[x]['wspan'][1]+1]) for x in t_depth_by_spans if t_depth_by_spans[x] == depth]))
        output.append('      Depth = %d counts by span length (words):\n' %depth)
        for span_length in sorted(depth_length_counts_sent[depth].keys()):
          output.append('        D%d length %d count = %d\n' %(depth, span_length, len(depth_length_counts_sent[depth][span_length])))
          for s in depth_length_counts_sent[depth][span_length]:
            output.append('          %s\n' %s)
      output.append('\n')
      output.append('  Num gold constituents: %d\n' %len(g_spans_complex))
      output.append('  Num test constituents: %d\n' %len(t_spans_complex))
      output.append('  Matching: %d\n' %len(brac_same))
      output.append('  False positives (in test but not gold): %d\n' %brac_t_only)
      output.append('  False negatives (in gold but not test): %d\n' %brac_g_only)
      output.append('  Precision: %.4f\n' %brac_p)
      output.append('  Recall: %.4f\n' %brac_r)
      output.append('  F-Measure: %.4f\n' %brac_f1)
      output.append('\n')
      output.append(str(debug_maxproj) + '\n')

    sentid += 1

    # Read the next pair of trees
    try:
      g_line = next(gold)
    except StopIteration:
      g_line = None
    while g_line and g_line.strip() == '':
      try:
        g_line = next(gold)
      except StopIteration:
        g_line = None

    try:
      t_line = next(test)
    except StopIteration:
      t_line = None
    while t_line and t_line.strip() == '':
      try:
        t_line = next(test)
      except StopIteration:
        t_line = None

  # Get overall segmentation accuracy scores
  seg_p, seg_r, seg_f1 = accuracy(seg_tp, seg_fp, seg_fn)
  brac_p, brac_r, brac_f1 = accuracy(brac_tp, brac_fp, brac_fn)
  all_p, all_r, all_f1 = accuracy(all_tp, all_fp, all_fn)

  # Get part of speech accuracy scores
  pos_m2one = m21(leaves_t2g)
  pos_one2m = m21(leaves_g2t)
  pos_h = homog(leaves_t2g, leaves_g2t, n_wrds) 
  pos_c = homog(leaves_g2t, leaves_t2g, n_wrds)
  pos_vm = (2 * pos_h * pos_c) / (pos_h + pos_c)

  # Get constituent labeling accuracy scores
  lab_m2one = m21(same_t2g)
  lab_m2one = m21(same_g2t)
  lab_h = homog(same_t2g, same_g2t, brac_tp)
  lab_c = homog(same_g2t, same_t2g, brac_tp)
  lab_vm = (2 * lab_h * lab_c) / (lab_h + lab_c)

  if 'NP' in same_g2t:
    # Get NP prediction scores.
    # Start with overlapping bracketings...
    test_np = None
    test_np_count = 0
    total_same = 0
    test_np = max(t_cat_counts, key=lambda x: t_cat_counts[x])
    test_np_count = t_cat_counts[test_np]
    np_predict_tp = same_t2g[test_np]['NP']
    np_predict_fn = max_proj_scores['NP']['hit'] + max_proj_scores['NP']['miss'] - np_predict_tp
    np_predict_fp = t_cat_counts[test_np] - np_predict_tp
    np_predict_p, np_predict_r, np_predict_f1 = accuracy(np_predict_tp, np_predict_fp, np_predict_fn)

  # Print final evaluation scores
  output.append('\n')
  output.append('######################################\n')
  output.append('Corpus-wide eval results\n')
  output.append('######################################\n')
  output.append('\n')
  output.append('Minimum constituent length = %d\n' %minlength)
  output.append('\n')
  output.append('Overall (character spans, includes segmentation and bracketing):\n')
  output.append('  Precision: %.4f\n' %all_p)
  output.append('  Recall: %.4f\n' %all_r)
  output.append('  F-Measure: %.4f\n' %all_f1)
  output.append('\n')
  output.append('Segmentation:\n')
  output.append('  Precision: %.4f\n' %seg_p)
  output.append('  Recall: %.4f\n' %seg_r)
  output.append('  F-Measure: %.4f\n' %seg_f1)
  output.append('\n')
  output.append('Bracketing:\n')
  output.append('  Precision: %.4f\n' %brac_p)
  output.append('  Recall: %.4f\n' %brac_r)
  output.append('  F-Measure: %.4f\n' %brac_f1)
  output.append('\n')
  output.append('Tagging accuracy:\n')
  output.append('  Many-to-1 accuracy: %.4f\n' %pos_m2one)
  output.append('  1-to-many accuracy: %.4f\n' %pos_one2m)
  output.append('  Homogeneity: %.4f\n' %pos_h)
  output.append('  Completeness: %.4f\n' %pos_c)
  output.append('  V-Measure: %.4f\n' %pos_vm)
  output.append('\n')
  output.append('Labeling accuracy of correct constituents:\n')
  output.append('  Many-to-1 accuracy: %.4f\n' %lab_m2one)
  output.append('  1-to-many accuracy: %.4f\n' %lab_m2one)
  output.append('  Homogeneity: %.4f\n' %lab_h)
  output.append('  Completeness: %.4f\n' %lab_c)
  output.append('  V-Measure: %.4f\n' %lab_vm)
  output.append('\n')
  if 'NP' in same_g2t:
    output.append('NP prediction accuracy -- map most frequent test label ("%s") to "NP":\n' %test_np)
    output.append('  Precision: %.4f\n' %np_predict_p)
    output.append('  Recall: %.4f\n' %np_predict_r)
    output.append('  F-Measure: %.4f\n' %np_predict_f1)
    output.append('\n')
  output.append('Depth counts:\n')
  for depth in depth_counts:
    output.append('  Depth = %d: %d\n' %(depth, depth_counts[depth]))
    output.append('    Depth = %d counts by span length (words):\n' %depth)
    for span_length in sorted(depth_length_counts[depth].keys()):
      output.append('      D%d length %d count = %d\n' %(depth, span_length, depth_length_counts[depth][span_length]))
    output.append('\n')
  output.append('Maximal projection identification accuracy:\n')
  output.append('\n')
  for cat in max_proj:
    if max_proj_scores[cat]['hit'] + max_proj_scores[cat]['miss'] > 0:
      output.append('Type %s\n' %cat)
      output.append('  Found: %d\n' %max_proj_scores[cat]['hit'])
      output.append('  Not found: %d\n' %max_proj_scores[cat]['miss'])
      output.append('  % ' + cat + ' constituents identified: %.4f\n' %(float(max_proj_scores[cat]['hit']) / float(max_proj_scores[cat]['hit'] + max_proj_scores[cat]['miss'])))
      if cat in same_g2t:
        output.append('  Labeling of found ' + cat + ' constituents:\n')
        total = max_proj_scores[cat]['hit']
        for label in same_g2t[cat]:
          output.append('    Label ' + label + ': %.2f\n' %(float(same_g2t[cat][label])/float(total) * 100))
        output.append('\n')
    else:
      output.append('  No instances of %s found in gold trees\n' %cat)
      output.append('\n')

  return output
