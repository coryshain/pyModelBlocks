from .linetrees import *


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


class LineTreesNatstorPennSource(ExternalResource):
    DEFAULT_LOCATION = 'parses/penn/all-parses.txt.penn'
    STATIC_PREREQ_TYPES = [NatstorRepo]
    PARENT_RESOURCE = NatstorRepo
    DESCR_SHORT = 'Natural Stories PTB source trees'
    DESCR_LONG = 'Source Penn Treebank style hand-corrected parses for the Natural Stories corpus'


class LineTreesNatstorPTB(LineTrees):
    MANIP = 'naturalstories.penn'
    STATIC_PREREQ_TYPES = [LineTreesNatstorPennSource, ScriptsEditabletrees2linetrees]
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
                    out = out.replace(
                        '``', "'"
                    ).replace(
                        "''", "'"
                    ).replace(
                        "(",
                        "-LRB-"
                    ).replace(
                        ")",
                        "-RRB-"
                    ).replace(
                        "peaked",
                        "peeked"
                    )

                    outputs.append(out)

            return outputs

        return out

    @classmethod
    def other_prereq_paths(self, path):
        directory = os.path.dirname(path)
        filename = 'naturalstories.penn.linetrees'

        return [os.path.join(directory, filename)]
