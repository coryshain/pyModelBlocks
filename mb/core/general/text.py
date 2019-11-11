from .core import *


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





#####################################
#
# LINETOKS TYPES
#
#####################################


class LineToksMorphed(LineToks):
    MANIP = '.morph'
    STEM_PREREQ_TYPES = [LineToks]
    DESCR_SHORT = 'morphed linetoks'
    DESCR_LONG = 'Run morfessor over linetoks'

    def body(self):
        out = 'morfessor -t %s -T %s --output-format={analysis}' ' --output-newlines > %s' % (
            self.stem_prereqs()[0].path,
            self.stem_prereqs()[0].path,
            self.path,
        )

        return out


class LineToksDelim(LineToks):
    MANIP = '.delim'
    STEM_PREREQ_TYPES = [LineToks]
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
    STEM_PREREQ_TYPES = [LineToks]
    DESCR_SHORT = 'reversed linetoks'
    DESCR_LONG = 'Reverse linetoks'

    def body(self):
        def out(inputs):
            outputs = []
            for x in inputs:
                outputs.append(' '.join(x.strip().split()[::-1]) + '\n')

            return outputs

        return out
