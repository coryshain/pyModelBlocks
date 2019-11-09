from mb.core.regression.regression import *
from mb.external_resources.cdr import CDRRepo

#####################################
#
# REGRESSION EXECUTABLE TYPES
#
#####################################


class RegressionExecutableCDR(RegressionExecutable):
    MANIP = 'regress-cdr'
    STATIC_PREREQ_TYPES = ['scripts/regress-cdr.sh', CDRRepo]
    DESCR_SHORT = 'CDR regression executable'
    DESCR_LONG = "Exectuable for fitting a continuous-time deconvolutional regression (CDR) model\n"

    def body(self):
        out = 'cp %s %s' % (
            self.static_prereqs()[0].path,
            self.path
        )

        return out





#####################################
#
# PREDICTION EXECUTABLE TYPES
#
#####################################


class PredictionExecutableCDR(PredictionExecutable):
    MANIP = 'predict-cdr'
    STATIC_PREREQ_TYPES = ['scripts/predict-cdr.sh', CDRRepo]
    DESCR_SHORT = 'CDR prediction executable'
    DESCR_LONG = "Exectuable for prediction from a continuous-time deconvolutional regression (CDR) model\n"

    def body(self):
        out = 'cp %s %s' % (
            self.static_prereqs()[0].path,
            self.path
        )

        return out





#####################################
#
# REGRESSION TYPES
#
#####################################


class RegressionCDR(Regression):
    REGRESSION_TYPE = 'cdr'
    DESCR_SHORT = 'CDR regression'
    DESCR_LONG = "Run continuous-time deconvolutional regression (LMER)\n"





#####################################
#
# PREDICTION TYPES
#
#####################################


class PredictionCDR(Prediction):
    REGRESSION_TYPE = 'cdr'
    DESCR_SHORT = 'CDR prediction'
    DESCR_LONG = "Predict from continuous-time deconvolutional regression (LMER)\n"


