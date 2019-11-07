from mb.core.tables import *




#####################################
#
# ABSTRACT TYPES
#
#####################################


class Regression(MBType):
    SUFFIX = '.regression'
    PATTERN_PREREQ_TYPES = [EvMeasures]
    ARG_TYPES = [
        Arg(
            'model_config_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing model configuration instructions.'
        ),
        Arg(
            'predictors',
            dtype=str,
            positional=True,
            descr='Underscore-delimited list of predictors to add to baseline formula defined in the model config.'
        ),
        Arg(
            'partition_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing partitioning instructions.'
        ),
        Arg(
            'partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use. One of ["fit", "expl", "held"].'
        ),
        Arg(
            'c',
            dtype=str,
            positional=False,
            descr='Basename of *.ini file in local directory ``prm`` providing censorship instructions.'
        )
    ]
    FILE_TYPE = None
    PRECIOUS = True
    REGRESSION_TYPE = ''
    DESCR_SHORT = 'regression'
    DESCR_LONG = "Abstract base class for regression types\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'Regression'

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return []

        args = cls.parse_args(path)

        directory = os.path.dirname(path)
        resmeasures = os.path.basename(path).split('.')[0]
        resmeasures += '.%s-%s' % (args['partition_params_file'], args['partition_name'])

        c = args['c']
        if c is not None:
            resmeasures += '-c%s' % c

        resmeasures += '.resmeasures'

        out = [
            'bin/regress-%s' % cls.regression_type(),
            os.path.join(directory, resmeasures),
            'prm/%s.regprm.ini' % args['model_config_file']
        ]
            
        return out

    @classmethod
    def regression_type(cls):
        return cls.REGRESSION_TYPE

    def body(self):
        preds = self.args['predictors'].split(DELIM[2])

        evmeasures = self.pattern_prereqs()[0]
        executable, resmeasures, config = self.other_prereqs()

        out = '%s %s %s %s %s  >  %s  2>  %s.log' % (
            executable.path,
            evmeasures.path,
            resmeasures.path,
            config.path,
            ' '.join(preds),
            self.path,
            self.path
        )

        return out


class Regress(MBType):
    DESCR_SHORT = 'regression executable'
    DESCR_LONG = "Abstract base class for regression executables\n"




#####################################
#
# REGRESSION TYPES
#
#####################################


class LMER(Regression):
    MANIP = '-lmer'
    REGRESSION_TYPE = 'lmer'
    STATIC_PREREQ_TYPES = [ScriptsLmertools, ScriptsRegresslmer]
    DESCR_SHORT = 'LMER regression'
    DESCR_LONG = "Run linear mixed-effects (LMER) regression\n"






#####################################
#
# REGRESSION EXECUTABLE TYPES
#
#####################################




class RegressLMER(Regress):
    MANIP = 'regress-lmer'
    STATIC_PREREQ_TYPES = [ScriptsRegresslmer]
    DESCR_SHORT = 'LMER executable'
    DESCR_LONG = "Exectuable for running a linear mixed-effects (LMER) regression\n"

    def body(self):
        out = 'cp %s %s' % (
            self.static_prereqs()[0].path,
            self.path
        )

        return out
