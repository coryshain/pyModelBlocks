from mb.core.general.core import *

class CDRRepo(ExternalResource):
    URL = 'https://github.com/coryshain/cdr'
    DESCR_SHORT = 'Continuous-Time Deconvolutional Regression'
    DESCR_LONG = 'A repository for estimating temporally diffuse effects from time series data.'

    def body(self):
        return 'git clone git@github.com:coryshain/cdr.git %s' % self.path

    @property
    def max_timestamp(self):
        max_timestamp = self.timestamp

        return max_timestamp