from .linetrees import *


def create_classes_from_ptb_dir(directory, name='WSJ'):
    out = []
    for sect in sorted(os.listdir(directory)):
        DEFAULT_LOCATION = os.path.join(directory, sect)
        descr = '%s section %s (source)' % (name, sect)
        src_class_name = '%sSection%sSrc' % (name, sect)
        def section(cls):
            return cls.SECTION

        attr_dict = {
            'PARENT_RESOURCE': PennTreebankRepo,
            'DEFAULT_LOCATION': DEFAULT_LOCATION,
            'SECTION': sect,
            'DESCR_SHORT': descr,
            'DESCR_LONG': descr,
            'section': section
        }
        new_class = type(src_class_name, (ExternalResource,), attr_dict)
        new_class.section = classmethod(section)
        globals()[src_class_name] = new_class

        descr = '%s section %s' % (name, sect)
        class_name = '%sSection%s' % (name, sect)
        MANIP = '%s%s' % (name.lower(), sect)
        def body(self):
            src_path = self.static_prereqs[0].path
            mrg = [os.path.join(src_path, p) for p in sorted(os.listdir(src_path)) if p.endswith('.mrg')]

            out = "cat %s | perl %s | awk '/^\s*\(/' > %s" % (' '.join(mrg), self.static_prereqs[1].path, self.path)

            return out

        attr_dict = {
            'MANIP': MANIP,
            'STATIC_PREREQ_TYPES': [globals()[src_class_name], ScriptsEditabletrees2linetrees],
            'SECTION': sect,
            'DESCR_SHORT': descr,
            'DESCR_LONG': descr,
            'body': body,
            'section': section
        }
        new_class = type(class_name, (LineTrees,), attr_dict)
        new_class.section = classmethod(section)
        out.append(new_class)
        globals()[class_name] = new_class

    return out


class PennTreebankRepo(ExternalResource):
    URL = 'https://catalog.ldc.upenn.edu/ldc99t42'
    DESCR_SHORT = 'the Penn Treebank (PTB)'
    DESCR_LONG = (
        'One million words of 1989 Wall Street Journal material annotated in Treebank II style.\n'
        'A small sample of ATIS-3 material annotated in Treebank II style.\n'
        'Switchboard tagged, dysfluency-annotated, and parsed text.\n'
        'A fully tagged version of the Brown Corpus.\n'
        'Brown parsed text.\n'
    )

    def __init__(
            self,
            dump=False
    ):
        super(PennTreebankRepo, self).__init__()


WSJ_SECTIONS = create_classes_from_ptb_dir(
    os.path.join(
        PennTreebankRepo.infer_paths()[0],
        'parsed',
        'mrg',
        'wsj'
    ),
    name='WSJ'
)

SWBD_SECTIONS = create_classes_from_ptb_dir(
    os.path.join(
        PennTreebankRepo.infer_paths()[0],
        'parsed',
        'mrg',
        'swbd'
    ),
    name='SWBD'
)

BROWN_SECTIONS = create_classes_from_ptb_dir(
    os.path.join(
        PennTreebankRepo.infer_paths()[0],
        'parsed',
        'mrg',
        'brown'
    ),
    name='Brown'
)


class PTBSections(LineTrees):
    DESCR_SHORT = 'Concatenated PTB sections'
    DESCR_LONG = 'Concatenated PTB sections starting with START and ending with END'

    @classmethod
    def match(cls, path):
        m = re.match('(.*)(wsj|swbd|brown)(.+)to(.+)%s' % cls.suffix(), path)
        if m and os.path.basename(m.group(1)) == '':
            return True
        return False

    @classmethod
    def path_parser(cls, path):
        prefix, corpus, start, end = re.match('(.*)(wsj|swbd|brown)(.+)to(.+)%s' % cls.suffix(), path).groups()
        return {
            'basename': os.path.normpath(prefix),
            'corpus': corpus,
            'start': start,
            'end': end
        }

    @classmethod
    def parse_path(cls, path):
        out = None
        path = os.path.normpath(path)
        if cls.match(path):
            parsed = cls.path_parser(path)
            prereq_types = cls.prereq_types(path)
            out = []
            for p in prereq_types:
                basename = parsed['basename']
                corpus = parsed['corpus']
                sect = p.section()
                out.append({
                    'basename': os.path.join(basename, corpus + sect)
                })
        return out

    @classmethod
    def prereq_types(cls, path=None):
        if path is None:
            sects = WSJ_SECTIONS + SWBD_SECTIONS + BROWN_SECTIONS
        else:
            parsed = cls.path_parser(path)

            s_ix = None
            e_ix = None

            if parsed['corpus'] == 'wsj':
                sects = WSJ_SECTIONS
            elif parsed['corpus'] == 'swbd':
                sects = SWBD_SECTIONS
            elif parsed['corpus'] == 'brown':
                sects = BROWN_SECTIONS
            else:
                raise ValueError('Unrecognized PTB corpus: %s' % parsed['corpus'])

            for i in range(len(sects)):
                if sects[i].section() == parsed['start']:
                    s_ix = i
                    break
            assert s_ix is not None, 'PTB section %s not found.' % parsed['start']

            for i in range(len(sects) - 1, -1, -1):
                if sects[i].section() == parsed['end']:
                    e_ix = i
                    break
            assert s_ix is not None, 'PTB section %s not found.' % parsed['end']

            sects = sects[s_ix:e_ix + 1]

        return sects

    @classmethod
    def syntax_str(cls):
        return '(<DIR>/){wsj,swbd,brown}<START>to<END>.linetrees'

    def body(self):
        def out():
            outputs = []
            for x in self.prereqs:
                outputs += x.data
            return outputs
        return out
