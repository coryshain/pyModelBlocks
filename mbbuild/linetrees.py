from .core import *
from . import tree

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
# IMPLEMENTED TYPES
#
#####################################

class Evalb(MBType):
    MANIP = 'evalb'
    STATIC_PREREQ_TYPES = [SrcEvalb]
    FILE_TYPE = None

    def body(self):
        return 'gcc -Wall -g -o %s %s' % (self.path, self.static_prereqs()[0].path)


class Indent(MBType):
    MANIP = 'indent'
    STATIC_PREREQ_TYPES = [Rvtl, SrcIndent]
    CONFIG_KEYS = [('cflags', '-DNDEBUG -O3')]
    FILE_TYPE = None

    def body(self):
        cflags = self.config_values()[0][2]

        return 'g++ -I%s -Wall %s -g -lm  %s  -o %s' % (
            os.path.abspath(self.static_prereqs()[0].path),
            cflags,
            self.static_prereqs()[1].path,
            self.path
        )


class LineTreesFromEditableTrees(LineTrees):
    PATTERN_PREREQ_TYPES = [EditableTrees]
    STATIC_PREREQ_TYPES = [ScriptsEditabletrees2linetrees]
    DESCR_SHORT = 'linetrees from editabletrees'
    DESCR_LONG = "Convert editabletrees into linetrees.\n"

    def body(self):
        out = "cat %s  |  perl -pe 's/^[0-9]*://;'  |  perl %s  >  %s" % (
            self.pattern_prereqs()[0].path,
            self.static_prereqs()[0].path,
            self.path
        )

        return out


class EditableTreesFromLineTrees(EditableTrees):
    MANIP = '.fromlinetrees'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'linetrees from editabletrees'
    DESCR_LONG = "Convert editabletrees into linetrees.\n"

    @classmethod
    def other_prereq_paths(cls, path):
        return ['bin/indent']


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
    STATIC_PREREQ_TYPES = [ScriptsMaketreesnumbered]
    DESCR_SHORT = 'numbered editabletrees'
    DESCR_LONG = "Add line numbers to editable trees.\n"

    def body(self):
        out = "cat %s | perl %s  >  %s" % (
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
                    outputs.append(str(t))

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
                    outputs.append(str(t))
                
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
    ARG_TYPES = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'first N linetrees'
    DESCR_LONG = "Truncate linetrees file to contain the first N lines.\n"

    def body(self):
        def out(inputs, n):
            return inputs[:n]

        return out


class LineTreesOnward(LineTrees):
    MANIP = 'onward'
    PATTERN_PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'linetrees from N onward'
    DESCR_LONG = "Truncate linetrees file to remove the first N-1 lines.\n"

    def body(self):
        def out(inputs, n):
            return inputs[n:]

        return out


class LineTreesLast(LineTrees):
    MANIP = 'last'
    PATTERN_PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [Arg('n', dtype=int, positional=True)]
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
    STATIC_PREREQ_TYPES = [ScriptsMaketreesnounary]
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


class LineTreesMerged(LineTrees):
    MANIP = '.merged'
    PATTERN_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'merged linetrees'
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


class LineTreesMergedPrefix(LineTreesMerged):
    HAS_PREFIX = True







