from .text import *
from mb.util.tabular import roll_toks, augment_cols, merge_tables, censor, partition


#####################################
#
# ABSTRACT TYPES
#
#####################################


class TokMeasures(MBType):
    SUFFIX = '.tokmeasures'
    FILE_TYPE = 'table'
    DESCR_SHORT = 'tokmeasures'
    DESCR_LONG = "Abstract base class for token-by-token measures (tokmeasures) types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'TokMeasures'


class ItemMeasures(MBType):
    SUFFIX = '.itemmeasures'
    FILE_TYPE = 'table'
    DESCR_SHORT = 'itemmeasures'
    DESCR_LONG = "Abstract base class for item-by-item measures (itemmeasures) types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'ItemMeasures'


class EvMeasures(MBType):
    SUFFIX = '.evmeasures'
    FILE_TYPE = 'table'
    DESCR_SHORT = 'evmeasures'
    DESCR_LONG = "Abstract base class for event measures (evmeasures) types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'EvMeasures'


class PrdMeasures(MBType):
    SUFFIX = '.prdmeasures'
    FILE_TYPE = 'table'
    DESCR_SHORT = 'prdmeasures'
    DESCR_LONG = "Abstract base class for predictor measures (prdmeasures) types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'PrdMeasures'


class ResMeasures(MBType):
    SUFFIX = '.resmeasures'
    FILE_TYPE = 'table'
    DESCR_SHORT = 'resmeasures'
    DESCR_LONG = "Abstract base class for response measures (resmeasures) types.\n"

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'ResMeasures'





#####################################
#
# TOKMEASURES TYPES
#
#####################################


class TokMeasuresDLT(TokMeasures):
    MANIP = '.dlt'
    PATTERN_PREREQ_TYPES = [GoldLineTrees]
    STATIC_PREREQ_TYPES = [ScriptsDlt_py]
    DESCR = 'DLT measures'
    DESCR_LONG = 'Compute DLT (integration cost) measures from linetrees'

    @classmethod
    def augment_prereq(cls, i, path):
        return '.gold'

    def body(self):
        out = "cat %s | python3 -m mb.static_resources.scripts.dlt > %s" % (
            self.pattern_prereqs()[0].path,
            self.path
        )

        return out





#####################################
#
# ITEMMEASURES TYPES
#
#####################################


class ItemmeasuresRolled(ItemMeasures):
    PATTERN_PREREQ_TYPES = [TokMeasures]
    DESCR_SHORT = 'rolled itemmeasures'
    DESCR_LONG = 'Itemmeasures rolled from tokmeasures'

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)<LINEITEMS>.lineitems']
        basename = '.'.join(path.split('.')[:-2])
        out = [basename + '.itemmeasures']

        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        return ItemMeasures

    def body(self):
        def out(tokmeasures, lineitems):
            outputs = roll_toks(tokmeasures, lineitems, skip_cols=['sentid', 'embddepthMin', 'timestamp'])

            return outputs

        return out


class ItemmeasuresConcat(ItemMeasures):
    MANIP = '.concat'
    PATTERN_PREREQ_TYPES = [ItemMeasures]
    REPEATABLE_PREREQ = True
    DESCR_SHORT = 'concatenated itemmeasures'
    DESCR_LONG = 'Itemmeasures from (column) concatenation of tables'

    def body(self):
        def out(*args):
            same_rows = True
            n_rows = None
            for x in args:
                if n_rows is None:
                    n_rows = x.shape[0]
                else:
                    if n_rows != x.shape[0]:
                        same_rows = False
                        break

            assert same_rows, 'All inputs must have the same number of rows. Got: %s.' % [x.shape[0] for x in args]

            args = [x.reset_index(drop=True) for x in args]

            colset = set()
            coldrop = {'word', 'sentpos', 'sentid', 'docid', 'rolled'}

            def col_mapper(x):
                if x in colset:
                    i = 1
                    colname = x + str(i)
                    while colname in colset:
                        i += 1
                    out = colname
                else:
                    out = x

                return out


            new_args = []
            for i, x in enumerate(args):
                coldrop_cur = colset & coldrop
                x = x[[y for y in x.columns if not y in coldrop_cur]]
                x = x.rename(col_mapper, axis=1)
                colset |= set(x.columns)
                new_args.append(x)

            outputs = pd.concat(new_args, axis=1)
            
            outputs = augment_cols(outputs)

            return outputs

        return out





#####################################
#
# EVMEASURES TYPES
#
#####################################


class EvMeasuresMerged(EvMeasures):
    PATTERN_PREREQ_TYPES = [ItemMeasures]
    DESCR_SHORT = 'merged evmeasures'
    DESCR_LONG = 'Merge of evmeasures with itemmeasures'

    @classmethod
    def augment_prereq(cls, i, path):
        return '.concat'

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)<CORPUS>.evmeasures']
        base = os.path.basename(path).split('.')[0]
        out = [os.path.join(os.path.dirname(path), '%s.evmeasures' % base)]

        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        return EvMeasures

    def body(self):
        def out(itemmeasures, evmeasures):
            itemmeasures = itemmeasures.copy()
            evmeasures = evmeasures.copy()
            outputs = merge_tables(evmeasures, itemmeasures, ['sentid', 'sentpos'])
            return outputs

        return out





#####################################
#
# PRDMEASURES TYPES
#
#####################################




#####################################
#
# RESMEASURES TYPES
#
#####################################


class ResMeasuresReg(ResMeasures):
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
            'partition_name',
            dtype=str,
            positional=True,
            descr='Name of partition element to use. One of ["fit", "expl", "held"].'
        )
    ]
    DESCR_LONG = 'resmeasures for regression analysis'

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['prm/<CENS-PARAMS>.partprm.ini', 'prm/<CENS-PARAMS>.censprm.ini']
        
        out = []
        args = cls.parse_args(path)

        out.append('prm/%s.partprm.ini' % args['part_params_file'])
        out.append('prm/%s.censprm.ini' % args['cens_params_file'])

        return out

    @classmethod
    def other_prereq_type(cls, i, path):
        return ParamFile

    def body(self):
        def out(*args):
            evmeasures = args[0].copy()
            other_prereqs = self.other_prereqs()
            part_params_file = other_prereqs[0].path
            cens_params_file = other_prereqs[1].path

            part_name = self.args['partition_name'].split(DELIM[2])

            evmeasures = censor(evmeasures, cens_params_file)
            evmeasures = partition(evmeasures, part_params_file, part_name)

            return evmeasures

        return out





