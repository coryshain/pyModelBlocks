from .core import *
from mb.util import tree
from mb.util.deps2trees import deps2trees
from mb.util.rules2headmodel import rules2headmodel
from mb.util.trees2deps import trees2deps
from mb.util.tree_compare import compare_trees
from mb.util.constit_eval import constit_eval
from mb.util.plug_leaves import plug_leaves


#####################################
#
# ABSTRACT TYPES
#
#####################################

class LineToks(MBType):
    SUFFIX = '.linetoks'
    DESCR_SHORT = 'linetoks'
    DESCR_LONG = "Abstract base class for linetoks types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'LineToks'


class LineItems(MBType):
    SUFFIX = '.lineitems'
    DESCR_SHORT = 'lineitems'
    DESCR_LONG = "Abstract base class for lineitems types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'LineItems'


class LineTrees(MBType):
    SUFFIX = '.linetrees'
    DESCR_SHORT = 'linetrees'
    DESCR_LONG = "Abstract base class for linetrees types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'LineTrees'


class Rules(MBType):
    SUFFIX = '.rules'
    DESCR_SHORT = 'rules'
    DESCR_LONG = "Abstract base class for PCFG rules file types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'Rules'


class Model(MBType):
    SUFFIX = '.model'
    DESCR_SHORT = 'model'
    DESCR_LONG = "Abstract base class for PCFG model file types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'Model'


class EditableTrees(MBType):
    SUFFIX = '.editabletrees'
    DESCR_SHORT = 'editable trees'
    DESCR_LONG = "Abstract base class for editable (indented) tree types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'EditableTrees'


class TokDeps(MBType):
    SUFFIX = '.tokdeps'
    DESCR_SHORT = 'tokdeps'
    DESCR_LONG = "Abstract base class for tokdeps types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'TokDeps'




#####################################
#
# COMPILED BINARIES
#
#####################################

class Evalb(MBType):
    MANIP = 'evalb'
    STATIC_PREREQ_TYPES = ['src/evalb.c']
    FILE_TYPE = None

    def body(self):
        return 'gcc -Wall -g -o %s %s' % (self.path, self.static_prereqs()[0].path)


class Indent(MBType):
    MANIP = 'indent'
    STATIC_PREREQ_TYPES = ['src/rvtl', 'src/indent.cpp']
    CONFIG_KEYS = [('c_flags', USER_SETTINGS.get('c_flags', DEFAULT_SETTINGS['c_flags']))]
    FILE_TYPE = None

    def body(self):
        c_flags = self.config_values()[0][2]

        return 'g++ -I%s -Wall %s -g -lm  %s  -o %s' % (
            os.path.abspath(self.static_prereqs()[0].path),
            c_flags,
            self.static_prereqs()[1].path,
            self.path
        )





#####################################
#
# LINETREES TYPES
#
#####################################


class LineTreesFromEditableTrees(LineTrees):
    PATTERN_PREREQ_TYPES = [EditableTrees]
    STATIC_PREREQ_TYPES = ['scripts/editabletrees2linetrees.pl']
    DESCR_SHORT = 'linetrees from editabletrees'
    DESCR_LONG = "Convert editabletrees into linetrees.\n"

    def body(self):
        out = "cat %s  |  perl -pe 's/^[0-9]*://;'  |  perl %s  >  %s" % (
            self.pattern_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesUpper(LineTrees):
    MANIP = '.upper'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'uppercased linetrees'
    DESCR_LONG = "Uppercase words in trees.\n"

    def body(self):
        def out(inputs):
            t = tree.Tree()
            outputs = []
            for x in inputs:
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    t.upper()
                    outputs.append(str(t) + '\n')

            return outputs

        return out


class LineTreesLower(LineTrees):
    MANIP = '.lower'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'lowercased linetrees'
    DESCR_LONG = "Lowercase words in trees.\n"
    
    def body(self):
        def out(inputs):
            t = tree.Tree()
            outputs = []
            for x in inputs:
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    t.lower()
                    outputs.append(str(t) + '\n')
                
            return outputs

        return out


class LineTreesRightBranching(LineTrees):
    MANIP = '.rb'
    PATTERN_PREREQ_TYPES = [LineToks]
    DESCR_SHORT = 'right-branching trees'
    DESCR_LONG = "Construct right-branching trees over sentences.\n"

    def body(self):
        def out(inputs):
            t = tree.Tree()
            left = '(1 '
            right = ') '
            outputs = []
            for x in inputs:
                x = x.strip()
                if x != '':
                    words = x.split()
                    out = ''
                    for word in words[::-1]:
                        w = left + word + right
                        if out:
                            out = left + w + out + right
                        else:
                            out = w
                    t.read(out)
                    outputs.append(str(t) + '\n')

            return outputs

        return out


class LineTreesFirst(LineTrees):
    MANIP = 'first'
    PATTERN_PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [
        Arg('n', dtype=int, positional=True, descr='Number of initial lines to retain')
    ]
    DESCR_SHORT = 'first N linetrees'
    DESCR_LONG = "Truncate linetrees file to contain the first N lines.\n"

    def body(self):
        def out(inputs, n):
            return inputs[:n]

        return out


class LineTreesOnward(LineTrees):
    MANIP = 'onward'
    PATTERN_PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [
        Arg('n', dtype=int, positional=True, descr='Number of initial lines to drop')
    ]
    DESCR_SHORT = 'linetrees from N onward'
    DESCR_LONG = "Truncate linetrees file to remove the first N-1 lines.\n"

    def body(self):
        def out(inputs, n):
            return inputs[n:]

        return out


class LineTreesLast(LineTrees):
    MANIP = 'last'
    PATTERN_PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [
        Arg('n', dtype=int, positional=True, descr='Number of final lines to retain')
    ]
    DESCR_SHORT = 'last N linetrees'
    DESCR_LONG = "Truncate linetrees file to contain the last N lines.\n"

    def body(self):
        def out(inputs, n):
            return inputs[-n:]

        return out


class LineTreesMaxWords(LineTrees):
    MANIP = 'maxwords'
    PATTERN_PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'linetrees with N words or fewer'
    DESCR_LONG = (
        "Truncate linetrees file to remove trees with > N words.\n"
    )

    def body(self):
        def out(inputs, n):
            t = tree.Tree()
            outputs = []
            for x in inputs:
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    if len(t.words()) <= n:
                        outputs.append(x + '\n')

            return outputs

        return out


class LineTreesNoUnary(LineTrees):
    MANIP = '.nounary'
    PATTERN_PREREQ_TYPES = [LineTrees]
    STATIC_PREREQ_TYPES = ['scripts/make-trees-nounary.pl']
    DESCR_SHORT = 'no-unary linetrees'
    DESCR_LONG = "Collapse unary CFG expansions.\n"

    def body(self):
        out = 'cat %s | perl %s > %s' % (
            self.pattern_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesNoDashTags(LineTrees):
    MANIP = '.nodashtags'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'no dash tags linetrees'
    DESCR_LONG = (
        "Remove dash tags from linetrees.\n"
    )

    def body(self):
        def out(inputs):
            filter = re.compile('([^(]+)[-=][^ )]+ ')
            outputs = []
            for x in inputs:
                outputs.append(filter.sub('\\1 ', x))

            return outputs

        return out


class LineTreesNoPunc(LineTrees):
    MANIP = '.nopunc'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'de-punc-ed linetrees'
    DESCR_LONG = (
        "Remove punctuation from linetrees.\n"
    )

    def body(self):
        def out(inputs):
            def is_punc(x):
                return x.c in PUNC

            t = tree.Tree()
            outputs = []
            for i, x in enumerate(inputs):
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    t.prune(is_punc)
                    outputs.append(str(t) + '\n')

            return outputs

        return out


class LineTreesNoCurrency(LineTrees):
    MANIP = '.nocurr'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'de-currencied linetrees'
    DESCR_LONG = (
        "Remove currency tokens from linetrees.\n"
    )

    def body(self):
        def out(inputs):
            def is_curr(x):
                return x.c == '$'

            t = tree.Tree()
            outputs = []
            for i, x in enumerate(inputs):
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    t.prune(is_curr)
                    outputs.append(str(t) + '\n')

            return outputs

        return out


class LineTreesReplaceParens(LineTrees):
    MANIP = '.replparens'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'paren-replaced linetrees'
    DESCR_LONG = (
        "Replace parens with string tokens '-LRB-', '-RRB-'.\n"
    )

    def body(self):
        def out(inputs):
            mapper = {
                '(': '-LRB-',
                ')': '-RRB-'
            }

            def labelmap(x):
                return mapper.get(x, x)

            t = tree.Tree()
            outputs = []
            for i, x in enumerate(inputs):
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    t.mapLabels(labelmap)
                    outputs.append(str(t) + '\n')

            return outputs

        return out


class LineTreesNoTrace(LineTrees):
    MANIP = '.nt'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'linetrees with traces removed'
    DESCR_LONG = (
        "Remove PTB traces from linetrees.\n"
    )

    def body(self):
        out = r"cat %s | sed 's/(-DFL- \+E_S) *//g;s/  \+/ /g;s/\t/ /g;s/\([^ ]\)(/\1 (/g;s/_//g;s/-UNDERSCORE-//g;s/([^ ()]\+ \+\*[^ ()]*)//g;s/( *-NONE-[^ ()]\+ *[^ ()]* *)//g;s/([^ ()]\+ )//g;s/ )/)/g;s/( /(/g;s/  \+/ /g;' | awk '!/^\s*\(CODE/' > %s " %(
            self.pattern_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesNoSubCat(LineTrees):
    MANIP = '.nosubcat'
    PATTERN_PREREQ_TYPES = [LineTreesNoUnary]
    DESCR_SHORT = 'linetrees with traces removed'
    DESCR_LONG = (
        "Remove subcat information from linetrees.\n"
    )

    def body(self):
        out = r"cat %s | perl -pe 's/\(([^- ]+)-[^ ]+ ([^ \)]*)\)/\(\1 \2\)/g' > %s" %(
            self.pattern_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesFromDeps(LineTrees):
    MANIP = '.fromdeps'
    PATTERN_PREREQ_TYPES = [TokDeps]
    DESCR_SHORT = 'linetrees from dependencies'
    DESCR_LONG = (
        "Convert dependencies to linetrees using the Collins et al. (1999) algorithm.\n"
        "Resultant trees are as flat as possible (i.e. as agnostic as possible about"
        "internal structure when heads have multiple dependents)."
    )

    def body(self):
        def out(inputs):
            outputs = deps2trees(iter(inputs))

            return outputs

        return out


class LineToksFromLineTrees(LineToks):
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'linetoks from linetrees'
    DESCR_LONG = 'Extract words (terminals) from linetrees'

    def body(self):
        def out(inputs):
            t = tree.Tree()
            outputs = []
            for x in inputs:
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    t.read(x)
                    outputs.append(' '.join(t.words()) + '\n')

            return outputs

        return out


class LineTreesConcat(LineTrees):
    MANIP = '.concat'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'concatenated linetrees'
    REPEATABLE_PREREQ = True
    DESCR_LONG = (
        "Concatenate linetrees files.\n"
    )

    def body(self):
        def out(*args):
            outputs = []
            for x in args:
                outputs += x
            return outputs

        return out





#####################################
#
# EDITABLETREES TYPES
#
#####################################


class EditableTreesFromLineTrees(EditableTrees):
    MANIP = '.fromlinetrees'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'linetrees from editabletrees'
    DESCR_LONG = "Convert editabletrees into linetrees.\n"

    @classmethod
    def other_prereq_paths(cls, path):
        return ['bin/indent']

    @classmethod
    def other_prereq_type(cls, i, path):
        return Indent

    def body(self):
        out = "cat %s | %s  >  %s" % (
            self.pattern_prereqs()[0].path,
            self.other_prereqs()[0].path,
            self.path
        )

        return out


class EditableTreesNumbered(EditableTrees):
    MANIP = '.numbered'
    PATTERN_PREREQ_TYPES = [EditableTrees]
    STATIC_PREREQ_TYPES = ['scripts/make-trees-numbered.pl']
    DESCR_SHORT = 'numbered editabletrees'
    DESCR_LONG = "Add line numbers to editable trees.\n"

    def body(self):
        out = "cat %s | perl %s  >  %s" % (
            self.pattern_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out





#####################################
#
# LINETOKS TYPES
#
#####################################


class LineToksMorphed(LineToks):
    MANIP = '.morph'
    PATTERN_PREREQ_TYPES = [LineToks]
    DESCR_SHORT = 'morphed linetoks'
    DESCR_LONG = 'Run morfessor over linetoks'

    def body(self):
        out = 'morfessor -t %s -T %s --output-format={analysis}' ' --output-newlines > %s' % (
            self.pattern_prereqs()[0].path,
            self.pattern_prereqs()[0].path,
            self.path,
        )

        return out


class LineToksDelim(LineToks):
    MANIP = '.delim'
    PATTERN_PREREQ_TYPES = [LineToks]
    DESCR_SHORT = 'delimited linetoks'
    DESCR_LONG = 'Add "!ARTICLE" delimiter to start of linetoks, if not already present'

    def body(self):
        def out(inputs):
            if len(inputs) == 0 or inputs[0].strip() != '!ARTICLE':
                inputs.insert(0, '!ARTICLE\n')

            return inputs

        return out


class LineToksReversed(LineToks):
    MANIP = '.rev'
    PATTERN_PREREQ_TYPES = [LineToks]
    DESCR_SHORT = 'reversed linetoks'
    DESCR_LONG = 'Reverse linetoks'

    def body(self):
        def out(inputs):
            outputs = []
            for x in inputs:
                outputs.append(' '.join(x.strip().split()[::-1]) + '\n')

            return outputs

        return out





#####################################
#
# PCFG RULES TYPES
#
#####################################


class RulesFromLineTrees(Rules):
    PATTERN_PREREQ_TYPES = [LineTrees]
    STATIC_PREREQ_TYPES = ['scripts/trees2rules.pl']
    DESCR_SHORT = 'rules from linetrees'
    DESCR_LONG = 'Convert linetrees into table of CFG rules'

    def body(self):
        out = 'cat %s | perl %s > %s' % (
            self.pattern_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )
        
        return out





#####################################
#
# PCFG MODEL TYPES
#
#####################################


class ModelFromRules(Model):
    PATTERN_PREREQ_TYPES = [Rules]
    DESCR_SHORT = 'model from rules'
    DESCR_LONG = 'Compute PCFG from table of CFG rules'

    def body(self):
        out = (
            """cat %s | sort | uniq -c | sort -nr | awk '{"wc -l %s | cut -d\\" \\" -f1" | """
            """getline t; u = $1; $1 = u/t; print;}' | awk '{p = $1; for (i=1;i<NF;i++) $i=$(i+1);$NF="="; """
            """$(NF + 1)=p; tmp=$2;$2=$3;$3=tmp;$1="R";print;}' > %s"""
        ) % (
            self.pattern_prereqs()[0].path,
            self.pattern_prereqs()[0].path,
            self.path
        )

        return out


class ModelHead(Model):
    MANIP = '.head'
    PATTERN_PREREQ_TYPES = [Model]
    DESCR_SHORT = 'headmodel'
    DESCR_LONG = 'Compute headedness model (probs of head given parent) from a PCFG model'

    def body(self):
        def out(inputs):
            outputs = rules2headmodel(inputs)

            return outputs

        return out





#####################################
#
# SYNTACTIC EVALUATION TYPES
#
#####################################


class Syneval(MBType):
    SUFFIX = '.syneval'
    PATTERN_PREREQ_TYPES = [LineTrees, LineTrees]
    STATIC_PREREQ_TYPES = ['prm/evalb.prm']
    DESCR_SHORT = 'syneval'
    DESCR_LONG = "Evaluate one syntactic annotation (parse) against another.\n"

    def body(self):
        linetrees = self.pattern_prereqs()

        out = '%s -p %s %s %s > %s' % (
            self.other_prereqs()[0].path,
            self.static_prereqs()[0].path,
            linetrees[0].path,
            linetrees[1].path,
            self.path
        )

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        return ['bin/evalb']

    @classmethod
    def other_prereq_type(cls, i, path):
        return Evalb

    @classmethod
    def other_prereq_type(cls, i, path):
        if i == 0:
            return RegressionExecutable
        if i == 1:
            return ResMeasures
        if i == 2:
            return ParamFile
        raise TypeError(other_prereq_type_err_msg(i, 3))


class SynevalErrors(MBType):
    SUFFIX = '.syneval.errs'
    PATTERN_PREREQ_TYPES = [LineTrees, LineTrees]
    DESCR_SHORT = 'syneval errors'
    DESCR_LONG = "Report parse errors in trees (2nd arg) compared to gold (1st arg).\n"

    def body(self):
        def out(*args):
            gold = args[0]
            pred = args[1]

            outputs = compare_trees(gold, pred)

            return outputs

        return out


class ConstitEval(MBType):
    SUFFIX = '.constiteval'
    PATTERN_PREREQ_TYPES = [LineTrees, LineTrees]
    DESCR_SHORT = 'constituent eval (syneval plus)'
    DESCR_LONG = "Run constituent evaluation (suite of metrics for unsupervised parsing eval).\n"

    def body(self):
        def out(*args):
            gold = iter(args[0])
            pred = iter(args[1])

            outputs = constit_eval(gold, pred)

            return outputs

        return out


class ConstitEvalSuffix(ConstitEval):
    HAS_SUFFIX = True


class BootstrapSignif(MBType):
    SUFFIX = '.bootstrapsignif'
    PATTERN_PREREQ_TYPES = [Syneval, Syneval]
    STATIC_PREREQ_TYPES = ['scripts/compare.pl']
    DESCR_SHORT = 'syneval signif test'
    DESCR_LONG = "Statistically compare difference between two synevals by bootstrap test.\n"

    def body(self):
        synevals = self.pattern_prereqs()

        out = 'perl %s %s %s > %s' % (
            self.static_prereqs()[0].path,
            synevals[0].path,
            synevals[1].path,
            self.path
        )

        return out





#####################################
#
# TOKDEPS TYPES
#
#####################################


class TokDepsFromLineTrees(TokDeps):
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'tokdeps from linetrees'
    DESCR_LONG = 'Compute dependency representation from linetrees file, using empirically-derived head probabilities'

    def body(self):
        def out(*args):
            trees = args[0]
            headmodel = args[0]

            outputs = trees2deps(trees, headmodel)

            return outputs

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)<CORPUS>.head.model']
        return [cls.strip_suffix(path) + '.head.model']

    @classmethod
    def other_prereq_type(cls, i, path):
        return ModelHead





#####################################
#
# INFER GOLD TREES BY RESOURCE
#
#####################################


class GoldLineTrees(LineTrees):
    MANIP = '.gold'
    PATTERN_PREREQ_TYPES = [LineToks]
    DESCR = 'gold linetrees'
    DESCR_LONG = 'Human-annotated parses (linetrees)'

    def body(self):
        def out(words, trees):
            return plug_leaves(trees, words)

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)<CORPUS>.linetoks']
        dirname = os.path.join(ROOT_DIR, 'static_resources', 'annot')
        basename = cls.strip_suffix(os.path.basename(path))

        out = [os.path.join(dirname, basename + '.stripped.linetrees')]

        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        return StaticResource
