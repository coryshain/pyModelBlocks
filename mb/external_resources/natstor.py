from mb.core.general.tables import *
from mb.util.toks2sents import toks2sents
from mb.util.sents2sentids import sents2sentids
from mb.util.tabular import rt2timestamps
from mb.util.util_natstor import ns_text_normalizer, ns_docid_int2name, docids_by_item, textgrid2itemmeasures, ns_merge


#####################################
#
# EXTERNAL RESOURCES
#
#####################################


class NatstorRepo(ExternalResource):
    URL = 'https://github.com/languageMIT/naturalstories'
    DESCR_SHORT = 'the Natural Stories Corpus'
    DESCR_LONG = (
        'A corpus of naturalistic stories meant to contain varied,\n'
        'low-frequency syntactic constructions. There are a variety of annotations\n'
        'and psycholinguistic measures available for the stories.\n'
    )

    def body(self):
        return 'git clone git@github.com:languageMIT/naturalstories.git %s' % self.path

    @property
    def max_timestamp(self):
        max_timestamp = self.timestamp

        return max_timestamp


class LineTreesNatstorPennSource(ExternalResource):
    DEFAULT_LOCATION = 'parses/penn/all-parses.txt.penn'
    STATIC_PREREQ_TYPES = [NatstorRepo]
    PARENT_RESOURCE = NatstorRepo
    DESCR_SHORT = 'Natural Stories PTB source trees'
    DESCR_LONG = 'Source Penn Treebank style hand-corrected parses for the Natural Stories corpus'


class NatstorTokSource(ExternalResource):
    DEFAULT_LOCATION = 'naturalstories_RTS/all_stories.tok'
    STATIC_PREREQ_TYPES = [NatstorRepo]
    PARENT_RESOURCE = NatstorRepo
    FILE_TYPE = 'table'
    SEP = '\t'
    DESCR_SHORT = 'Natural Stories source tokenization'
    DESCR_LONG = 'Natural Stories source tokenization'


class NatstorProcessRTs(ExternalResource):
    DEFAULT_LOCATION = 'naturalstories_RTS/process_ns_mb.R'
    STATIC_PREREQ_TYPES = [NatstorRepo, ScriptsProcess_ns_mb_R]
    PARENT_RESOURCE = NatstorRepo
    FILE_TYPE = None
    DESCR_SHORT = 'RT processing script'
    DESCR_LONG = 'Forked processing script for Natural Stories RT data'

    def body(self):
        return 'cp %s %s' % (
            self.static_prereqs()[1].path,
            self.path
        )


class NatstorRTSource(ExternalResource):
    DEFAULT_LOCATION = 'naturalstories_RTS/processed_RTs_MB.tsv'
    STATIC_PREREQ_TYPES = [NatstorRepo, NatstorProcessRTs]
    PARENT_RESOURCE = NatstorRepo
    FILE_TYPE = 'table'
    SEP = '\t'
    DESCR_SHORT = 'naturalstories source RTs'
    DESCR_LONG = 'Natural Stories source reading times'

    def body(self):
        return 'cd %s; ./%s' % (
            os.path.join(self.static_prereqs()[0].path, 'naturalstories_RTS'),
            self.static_prereqs()[1].basename
        )


class NatstorAudioSource(ExternalResource):
    DEFAULT_LOCATION = 'audio'
    STATIC_PREREQ_TYPES = [NatstorRepo]
    PARENT_RESOURCE = NatstorRepo
    DESCR_SHORT = 'Natural Stories audio source'
    DESCR_LONG = 'Natural Stories audio source'


    def body(self):
        return 'cp %s %s' % (
            self.static_prereqs()[1].path,
            self.path
        )





#####################################
#
# NATURAL STORIES TYPES
#
#####################################


class LineTreesNatstorPTB(LineTrees):
    MANIP = 'naturalstories.ptb'
    STATIC_PREREQ_TYPES = [LineTreesNatstorPennSource, ScriptsEditabletrees2linetrees_pl]
    DESCR_SHORT = 'naturalstories gold ptb linetrees'
    DESCR_LONG = (
        "Hand-annotated parse trees for the Natural Stories corpus.\n"
    )

    def body(self):
        out = "cat %s | sed 's/\\r//g' | perl %s > %s" % (
            self.static_prereqs()[0].path,
            self.static_prereqs()[1].path,
            self.path
        )

        return out

class LineToksNatstor(LineToks):
    MANIP = 'naturalstories'
    DESCR_SHORT = 'naturalstories linetoks'
    DESCR_LONG = (
        'PTB Tokenized sentences (linetoks) for the Natural Stories corpus'
    )

    def body(self):
        def out(inputs):
            trace = re.compile('\*')

            t = tree.Tree()
            outputs = []
            for x in inputs:
                x = x.strip()
                if (x != '') and (x[0] != '%'):
                    # Extract words
                    t.read(x)
                    all_words = t.words()
                    out = ''
                    for w in all_words:
                        if not trace.match(w):
                            if out != '':
                                out += ' '
                            out += w
                    out += '\n'

                    # Normalize sentence
                    out = ns_text_normalizer(out)

                    outputs.append(out)

            return outputs

        return out

    @classmethod
    def other_prereq_paths(self, path):
        if path is None:
            return ['(DIR/)naturalstories.ptb.linetrees']

        directory = os.path.dirname(path)
        filename = 'naturalstories.ptb.linetrees'

        return [os.path.join(directory, filename)]

    @classmethod
    def other_prereq_type(cls, i, path):
        return LineTreesNatstorPTB


class LineItemsNatstor(LineItems):
    MANIP = 'naturalstories'
    STATIC_PREREQ_TYPES = [NatstorTokSource]
    DESCR = 'naturalstories lineitems'
    DESCR_LONG = 'Natural Stories lineitems'
    
    def body_args(self):
        out = self.static_prereqs() + self.other_prereqs()

        return out
    
    def body(self):
        def out(tokmeasures, linetoks):
            tokmeasures = tokmeasures.copy()
            tokmeasures.word = tokmeasures.word.map(ns_text_normalizer)

            outputs = toks2sents(linetoks, tokmeasures)

            return outputs

        return out


    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)naturalstories.linetoks']
        return [os.path.join(os.path.dirname(path), 'naturalstories.linetoks')]

    @classmethod
    def other_prereq_type(cls, i, path):
        return LineToksNatstor


class ItemMeasuresNatstor(ItemMeasures):
    MANIP = 'naturalstories'
    STATIC_PREREQ_TYPES = [NatstorTokSource]
    DESCR = 'naturalstories itemmeasures'
    DESCR_LONG = 'Natural Stories base itemmeasures'

    def body_args(self):
        out = self.static_prereqs() + self.other_prereqs()

        return out

    def body(self):
        def out(tokmeasures, lineitems):
            outputs = docids_by_item(lineitems, tokmeasures)

            return outputs

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)naturalstories.lineitems']
        return [os.path.join(os.path.dirname(path), 'naturalstories.lineitems')]

    @classmethod
    def other_prereq_type(cls, i, path):
        return LineItemsNatstor


class ItemMeasuresNatstorTime(ItemMeasures):
    MANIP = 'naturalstories.t'
    PATTERN_PREREQ_TYPES = [ItemMeasuresNatstor]
    STATIC_PREREQ_TYPES = [NatstorAudioSource]
    DESCR = 'naturalstories time itemmeasures'
    DESCR_LONG = 'Natural Stories timestamp itemmeasures'

    def body(self):
        def out(itemmeasures):
            outputs = textgrid2itemmeasures(itemmeasures, self.static_prereqs()[0].path)

            return outputs

        return out

    @classmethod
    def augment_prereq(cls, i, path):
        return 'naturalstories'


class ItemMeasuresNatstorMergeFields(ItemMeasures):
    MANIP = 'naturalstories.mfields'
    STATIC_PREREQ_TYPES = [NatstorTokSource]
    DESCR = 'naturalstories merge fields'
    DESCR_LONG = 'Natural Stories itemmeasures augmented with fields needed to merge with experimental measures (evmeasures)'

    def body_args(self):
        out = self.static_prereqs() + self.other_prereqs()

        return out

    def body(self):
        def out(tokmeasures, lineitems):
            sentids = sents2sentids(lineitems)
            sentids = sentids.reset_index(drop=True)

            tokmeasures = tokmeasures.copy()
            tokmeasures.word = tokmeasures.word.map(ns_text_normalizer)
            tokmeasures['docid'] = tokmeasures.item
            tokmeasures = ns_docid_int2name(tokmeasures)
            tokmeasures = tokmeasures.reset_index(drop=True)

            outputs = pd.concat([sentids, tokmeasures[['docid', 'item', 'zone']]], axis=1)

            return outputs

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)naturalstories.lineitems']
        return [os.path.join(os.path.dirname(path), 'naturalstories.lineitems')]

    @classmethod
    def other_prereq_type(cls, i, path):
        return LineItemsNatstor


class EvMeasuresNatStor(EvMeasures):
    MANIP = 'naturalstories'
    STATIC_PREREQ_TYPES = [NatstorRTSource]
    DESCR = 'naturalstories base evmeasures'
    DESCR_LONG = 'Natural Stories base evmesaures'

    def body_args(self):
        out = self.static_prereqs() + self.other_prereqs()

        return out

    def body(self):
        def out(evmeasures, itemmeasures):
            evmeasures = evmeasures.copy()
            evmeasures.word = evmeasures.word.map(ns_text_normalizer)
            evmeasures.rename(lambda x: 'subject' if x == 'WorkerId' else 'fdur' if x == 'RT' else x, axis=1, inplace=True)

            itemmeasures = itemmeasures.copy()

            outputs = ns_merge(evmeasures, itemmeasures)

            outputs = rt2timestamps(outputs)

            return outputs

        return out

    @classmethod
    def other_prereq_paths(cls, path):
        if path is None:
            return ['(DIR/)naturalstories.mfields.lineitems']
        return [os.path.join(os.path.dirname(path), 'naturalstories.mfields.itemmeasures')]

    @classmethod
    def other_prereq_type(cls, i, path):
        return ItemMeasuresNatstorMergeFields


