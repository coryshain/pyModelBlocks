from mb.core.general.tables import *


#####################################
#
# ABSTRACT TYPES
#
#####################################


class RegressionExecutable(MBType):
    PRECIOUS = True
    DESCR_SHORT = 'regression executable'
    DESCR_LONG = "Abstract base class for regression executables\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'RegressionExecutable'


class PredictionExecutable(MBType):
    PRECIOUS = True
    DESCR_SHORT = 'prediction executable'
    DESCR_LONG = "Abstract base class for prediction executables\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'PredictionExecutable'


class SignifExecutable(MBType):
    PRECIOUS = True
    DESCR_SHORT = 'signif test executable'
    DESCR_LONG = "Abstract base class for significance testing executables\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'SignifExecutable'


class Regression(MBType):
    SUFFIX = DELIM[0] + 'reg'
    PATTERN_PREREQ_TYPES = [EvMeasures]
    ARG_TYPES = [
        Arg(
            'cens_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing censorship instructions.'
        ),
        Arg(
            'part_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing partitioning instructions.'
        ),
        Arg(
            'train_partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use for training. One of ["fit", "expl", "held"].'
        ),
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
    def regression_type(cls):
        return cls.REGRESSION_TYPE

    @classmethod
    def manip(cls):
        return DELIM[1] + cls.regression_type()

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return [
                'bin/regress-%s' % cls.regression_type(),
                '(DIR/)<RESMEASURES>.resmeasures'
                'prm/<NAME>.%sprm.ini' % cls.regression_type()
            ]

        args = cls.parse_args(path)

        directory = os.path.dirname(path)
        resmeasures = os.path.basename(path).split(DELIM[0])[0]
        resmeasures += DELIM[0] + \
                       DELIM[1].join((
                           args['cens_params_file'],
                           args['part_params_file'],
                           args['train_partition_name']
                       )) + \
                       DELIM[0] + \
                       'resmeasures'

        out = [
            'bin/regress-%s' % cls.regression_type(),
            os.path.join(directory, resmeasures),
            'prm/%s.%sprm.ini' % (args['model_config_file'], cls.regression_type())
        ]
            
        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        if i == 0:
            return RegressionExecutable
        if i == 1:
            return ResMeasures
        if i == 2:
            return ParamFile
        raise TypeError(other_prereq_type_err_msg(i, 3))

    def body(self):
        preds = self.args['predictors'].split(DELIM[2])

        evmeasures = self.pattern_prereqs()[0]
        executable, resmeasures, config = self.other_prereqs()

        out = '%s %s %s %s %s %s  >  %s  2>  %s' % (
            executable.path,
            evmeasures.path,
            resmeasures.path,
            config.path,
            self.path,
            ' '.join(preds),
            '%s%ssummary' % (self.path, DELIM[0]),
            DELIM[0].join((self.path,'log'))
        )

        return out


class Prediction(MBType):
    SUFFIX = DELIM[0] + 'pred'
    PATTERN_PREREQ_TYPES = [Regression]
    ARG_TYPES = [
        Arg(
            'cens_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing censorship instructions.'
        ),
        Arg(
            'part_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing partitioning instructions.'
        ),
        Arg(
            'train_partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use for training. One of ["fit", "expl", "held"].'
        ),
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
            'eval_partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use for evaluation. One of ["fit", "expl", "held"].'
        ),
    ]
    FILE_TYPE = 'table'
    REGRESSION_TYPE = ''
    DESCR_SHORT = 'prediction'
    DESCR_LONG = "Abstract base class for prediction types\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'Prediction'

    @classmethod
    def regression_type(cls):
        return cls.REGRESSION_TYPE

    @classmethod
    def manip(cls):
        return DELIM[1] + cls.regression_type()

    @classmethod
    def augment_prereq(cls, i, path):
        args = cls.parse_args(path)
        out = DELIM[0] + \
              DELIM[1].join((
                  args['cens_params_file'],
                  args['part_params_file'],
                  args['train_partition_name'],
                  args['model_config_file'],
                  args['predictors']
              )) + \
              DELIM[1] + \
              cls.regression_type()

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return [
                'bin/predict-%s' % cls.regression_type(),
                '(DIR/)<EVMEASURES>.evmeasures'
                '(DIR/)<RESMEASURES>.resmeasures'
            ]

        args = cls.parse_args(path)

        directory = os.path.dirname(path)

        evmeasures = args['basename'] + DELIM[0] + 'evmeasures'

        resmeasures = os.path.basename(path).split(DELIM[0])[0]
        resmeasures += DELIM[0] + \
                       DELIM[1].join((
                           args['cens_params_file'],
                           args['part_params_file'],
                           args['eval_partition_name']
                       )) + \
                       DELIM[0] + \
                       'resmeasures'

        out = [
            'bin/predict-%s' % cls.regression_type(),
            evmeasures,
            os.path.join(directory, resmeasures)
        ]

        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        if i == 0:
            return PredictionExecutable
        if i == 1:
            return EvMeasures
        if i == 2:
            return ResMeasures
        raise TypeError(other_prereq_type_err_msg(i, 3))

    def body(self):
        reg = self.pattern_prereqs()[0]
        executable, evmeasures, resmeasures = self.other_prereqs()

        out = '%s %s %s %s  >  %s  2>  %s' % (
            executable.path,
            reg.path,
            evmeasures.path,
            resmeasures.path,
            self.path,
            DELIM[0].join((self.path,'log'))
        )

        return out


class Signif(MBType):
    SUFFIX = DELIM[0] + 'sig'
    PATTERN_PREREQ_TYPES = [Prediction]
    ARG_TYPES = [
        Arg(
            'cens_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing censorship instructions.'
        ),
        Arg(
            'part_params_file',
            dtype=str,
            positional=True,
            descr='Basename of *.ini file in local directory ``prm`` providing partitioning instructions.'
        ),
        Arg(
            'train_partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use for training. One of ["fit", "expl", "held"].'
        ),
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
            'eval_partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use for evaluation. One of ["fit", "expl", "held"].'
        ),
        Arg(
            'regression_type',
            dtype=str,
            positional=True,
            descr='Type of regression model to evaluate.'
        ),
    ]
    SIGNIF_TYPE = ''
    DESCR_SHORT = 'significance report'
    DESCR_LONG = "Abstract base class for significance report from model comparison\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'Signif'

    @classmethod
    def signif_type(cls):
        return cls.SIGNIF_TYPE

    @classmethod
    def manip(cls):
        return DELIM[1] + cls.signif_type()

    @classmethod
    def augment_prereq(cls, i, path):
        args = cls.parse_args(path)
        out = DELIM[0] + DELIM[1].join((
            args['cens_params_file'],
            args['part_params_file'],
            args['train_partition_name'],
            args['model_config_file'],
            args['predictors'],
            args['eval_partition_name'],
            args['regression_type']
        ))

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return [
                'bin/signif-%s' % cls.signif_type(),
                '(DIR/)<RESMEASURES>.resmeasures'
                'prm/<NAME>.<REG>prm.ini'
            ]

        def powerset(iterable):
            xs = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(xs, n) for n in range(len(xs) + 1))

        args = cls.parse_args(path)

        template = ''.join((
            args['basename'],
            DELIM[0],
            DELIM[1].join((
                args['cens_params_file'],
                args['part_params_file'],
                args['train_partition_name'],
                args['model_config_file'],
                '%s',
                args['eval_partition_name'],
                args['regression_type']
            )),
            DELIM[0],
            'pred'
        ))

        preds = args['predictors'].split(DELIM[2])
        pset = powerset(preds)

        out = []

        for s in pset:
            out_cur = []
            for p in preds:
                if p not in s:
                    out_cur.append('~%s' % p)
                else:
                    out_cur.append('%s' % p)
            out_cur = '_'.join(out_cur)
            if '~' in out_cur:
                out.append(template % out_cur)

        out = ['bin/signif-%s' % cls.signif_type()] + out

        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        if i == 0:
            return SignifExecutable
        else:
            return Prediction

    def body(self):
        other_prereqs = self.other_prereqs()
        executable = other_prereqs[0]
        preds = other_prereqs[1:] + self.pattern_prereqs()

        out = '%s %s  >  %s  2>  %s' % (
            executable.path,
            ' '.join([x.path for x in preds]),
            self.path,
            DELIM[0].join((self.path,'log'))
        )

        return out
