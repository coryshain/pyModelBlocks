from .text import *
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


class LineTrees(MBType):
    SUFFIX = '.linetrees'
    DESCR_SHORT = 'linetrees'
    DESCR_LONG = "Abstract base class for linetrees types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'LineTrees'


class EditableTrees(MBType):
    SUFFIX = '.editabletrees'
    DESCR_SHORT = 'editable trees'
    DESCR_LONG = "Abstract base class for editable (indented) tree types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'EditableTrees'





#####################################
#
# COMPILED BINARIES
#
#####################################


class Indent(MBType):
    MANIP = 'indent'
    STATIC_PREREQ_TYPES = ['src/rvtl', 'src/indent.cpp']
    CONFIG_KEYS = [('c_flags', USER_SETTINGS.get('c_flags', DEFAULT_SETTINGS['c_flags']))]
    FILE_TYPE = None
    PRECIOUS = True

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
    STEM_PREREQ_TYPES = [EditableTrees]
    STATIC_PREREQ_TYPES = ['scripts/editabletrees2linetrees.pl']
    DESCR_SHORT = 'linetrees from editabletrees'
    DESCR_LONG = "Convert editabletrees into linetrees.\n"

    def body(self):
        out = "cat %s  |  perl -pe 's/^[0-9]*://;'  |  perl %s  >  %s" % (
            self.stem_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesUpper(LineTrees):
    MANIP = '.upper'
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineToks]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
    STATIC_PREREQ_TYPES = ['scripts/make-trees-nounary.pl']
    DESCR_SHORT = 'no-unary linetrees'
    DESCR_LONG = "Collapse unary CFG expansions.\n"

    def body(self):
        out = 'cat %s | perl %s > %s' % (
            self.stem_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesNoDashTags(LineTrees):
    MANIP = '.nodashtags'
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'linetrees with traces removed'
    DESCR_LONG = (
        "Remove PTB traces from linetrees.\n"
    )

    def body(self):
        out = r"cat %s | sed 's/(-DFL- \+E_S) *//g;s/  \+/ /g;s/\t/ /g;s/\([^ ]\)(/\1 (/g;s/_//g;s/-UNDERSCORE-//g;s/([^ ()]\+ \+\*[^ ()]*)//g;s/( *-NONE-[^ ()]\+ *[^ ()]* *)//g;s/([^ ()]\+ )//g;s/ )/)/g;s/( /(/g;s/  \+/ /g;' | awk '!/^\s*\(CODE/' > %s " %(
            self.stem_prereqs()[0].path,
            self.path
        )

        return out


class LineTreesNoSubCat(LineTrees):
    MANIP = '.nosubcat'
    STEM_PREREQ_TYPES = [LineTreesNoUnary]
    DESCR_SHORT = 'linetrees with traces removed'
    DESCR_LONG = (
        "Remove subcat information from linetrees.\n"
    )

    def body(self):
        out = r"cat %s | perl -pe 's/\(([^- ]+)-[^ ]+ ([^ \)]*)\)/\(\1 \2\)/g' > %s" %(
            self.stem_prereqs()[0].path,
            self.path
        )

        return out


class LineToksFromLineTrees(LineToks):
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
    STEM_PREREQ_TYPES = [LineTrees]
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
            self.stem_prereqs()[0].path,
            self.other_prereqs()[0].path,
            self.path
        )

        return out


class EditableTreesNumbered(EditableTrees):
    MANIP = '.numbered'
    STEM_PREREQ_TYPES = [EditableTrees]
    STATIC_PREREQ_TYPES = ['scripts/make-trees-numbered.pl']
    DESCR_SHORT = 'numbered editabletrees'
    DESCR_LONG = "Add line numbers to editable trees.\n"

    def body(self):
        out = "cat %s | perl %s  >  %s" % (
            self.stem_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out





#####################################
#
# INFER GOLD TREES BY RESOURCE
#
#####################################


class GoldLineTrees(LineTrees):
    MANIP = '.gold'
    STEM_PREREQ_TYPES = [LineToks]
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