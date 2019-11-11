from ..general.tree import *





#####################################
#
# ABSTRACT TYPES
#
#####################################


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


class TokDeps(MBType):
    SUFFIX = '.tokdeps'
    DESCR_SHORT = 'tokdeps'
    DESCR_LONG = "Abstract base class for tokdeps types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'TokDeps'





#####################################
#
# Params
#
#####################################


class EvalbPrm(MBType):
    MANIP = 'evalb.prm'
    STATIC_PREREQ_TYPES = ['prm/evalb.prm']
    FILE_TYPE = None
    PRECIOUS = True

    def body(self):
        out = 'cp %s %s' % (self.static_prereqs()[0].path, self.path)

        return out





#####################################
#
# COMPILED BINARIES
#
#####################################


class Evalb(MBType):
    MANIP = 'evalb'
    STATIC_PREREQ_TYPES = ['src/evalb.c']
    FILE_TYPE = None
    PRECIOUS = True

    def body(self):
        return 'gcc -Wall -g -o %s %s' % (self.path, self.static_prereqs()[0].path)





#####################################
#
# LINETREES TYPES
#
#####################################


class LineTreesFromDeps(LineTrees):
    MANIP = '.fromdeps'
    STEM_PREREQ_TYPES = [TokDeps]
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





#####################################
#
# PCFG RULES TYPES
#
#####################################


class RulesFromLineTrees(Rules):
    STEM_PREREQ_TYPES = [LineTrees]
    STATIC_PREREQ_TYPES = ['scripts/trees2rules.pl']
    DESCR_SHORT = 'rules from linetrees'
    DESCR_LONG = 'Convert linetrees into table of CFG rules'

    def body(self):
        out = 'cat %s | perl %s > %s' % (
            self.stem_prereqs()[0].path,
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
    STEM_PREREQ_TYPES = [Rules]
    DESCR_SHORT = 'model from rules'
    DESCR_LONG = 'Compute PCFG from table of CFG rules'

    def body(self):
        out = (
            """cat %s | sort | uniq -c | sort -nr | awk '{"wc -l %s | cut -d\\" \\" -f1" | """
            """getline t; u = $1; $1 = u/t; print;}' | awk '{p = $1; for (i=1;i<NF;i++) $i=$(i+1);$NF="="; """
            """$(NF + 1)=p; tmp=$2;$2=$3;$3=tmp;$1="R";print;}' > %s"""
        ) % (
            self.stem_prereqs()[0].path,
            self.stem_prereqs()[0].path,
            self.path
        )

        return out


class ModelHead(Model):
    MANIP = '.head'
    STEM_PREREQ_TYPES = [Model]
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
    STEM_PREREQ_TYPES = [LineTrees, LineTrees]
    DESCR_SHORT = 'syneval'
    DESCR_LONG = "Evaluate one syntactic annotation (parse) against another.\n"

    def body(self):
        linetrees = self.stem_prereqs()
        bin, prm = self.other_prereqs()

        out = '%s -p %s %s %s > %s' % (
            bin.path,
            prm.path,
            linetrees[0].path,
            linetrees[1].path,
            self.path
        )

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        return [
            'bin/evalb',
            'prm/evalb.prm'
        ]

    @classmethod
    def other_prereq_type(cls, i, path):
        if i == 0:
            return Evalb
        if i == 1:
            return EvalbPrm
        raise TypeError(other_prereq_type_err_msg(i, 2))


class SynevalErrors(MBType):
    SUFFIX = '.syneval.errs'
    STEM_PREREQ_TYPES = [LineTrees, LineTrees]
    DESCR_SHORT = 'syneval errors'
    DESCR_LONG = "Report parse errors in trees (2nd arg) compared to gold (1st arg).\n"

    def body(self):
        def out(gold, pred):
            outputs = compare_trees(gold, pred)

            return outputs

        return out


class ConstitEval(MBType):
    SUFFIX = '.constiteval'
    STEM_PREREQ_TYPES = [LineTrees, LineTrees]
    DESCR_SHORT = 'constituent eval (syneval plus)'
    DESCR_LONG = "Run constituent evaluation (suite of metrics for unsupervised parsing eval).\n"

    def body(self):
        def out(gold, pred):
            outputs = constit_eval(gold, pred)

            return outputs

        return out


class ConstitEvalSuffix(ConstitEval):
    ALLOW_SHARED_SUFFIX = True


class BootstrapSignif(MBType):
    SUFFIX = '.bootstrapsignif'
    STEM_PREREQ_TYPES = [Syneval, Syneval]
    STATIC_PREREQ_TYPES = ['scripts/compare.pl']
    ALLOW_SHARED_SUFFIX = False
    DESCR_SHORT = 'syneval signif test'
    DESCR_LONG = "Statistically compare difference between two synevals by bootstrap test.\n"

    def body(self):
        synevals = self.stem_prereqs()

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
    STEM_PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'tokdeps from linetrees'
    DESCR_LONG = 'Compute dependency representation from linetrees file, using empirically-derived head probabilities'

    def body(self):
        def out(trees, headmodel):
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