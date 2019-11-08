from mb.core.regression.regression import *

#####################################
#
# REGRESSION EXECUTABLE TYPES
#
#####################################


class RegressionExecutableLMER(RegressionExecutable):
    MANIP = 'regress-lmer'
    STATIC_PREREQ_TYPES = [ScriptsRegresslmer_sh, ScriptsLmertools_R, ScriptsRegresslmer_R]
    DESCR_SHORT = 'LMER regression executable'
    DESCR_LONG = "Exectuable for fitting a linear mixed-effects (LMER) regression model"

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


class PredictionExecutableLMER(PredictionExecutable):
    MANIP = 'predict-lmer'
    STATIC_PREREQ_TYPES = [ScriptsPredictlmer_sh, ScriptsLmertools_R, ScriptsPredictlmer_R]
    DESCR_SHORT = 'LMER prediction executable'
    DESCR_LONG = "Exectuable for prediction from a linear mixed-effects (LMER) regression model"

    def body(self):
        out = 'cp %s %s' % (
            self.static_prereqs()[0].path,
            self.path
        )

        return out





#####################################
#
# SIGNIF EXECUTABLE TYPES
#
#####################################


class SignifExecutableLRT(SignifExecutable):
    MANIP = 'signif-lrt'
    STATIC_PREREQ_TYPES = [ScriptsSigniflrt_sh, ScriptsLmertools_R, ScriptsSigniflrt_R, ScriptsSignif_py]
    DESCR_SHORT = 'LRT signif executable'
    DESCR_LONG = "Exectuable for computing likelihood ratio test (LRT)"

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


class RegressionLMER(Regression):
    REGRESSION_TYPE = 'lmer'
    DESCR_SHORT = 'LMER regression'
    DESCR_LONG = "Run linear mixed-effects (LMER) regression"





#####################################
#
# PREDICTION TYPES
#
#####################################


class PredictionLMER(Prediction):
    REGRESSION_TYPE = 'lmer'
    DESCR_SHORT = 'LMER prediction'
    DESCR_LONG = "Predict from linear mixed-effects (LMER) regression"





#####################################
#
# SIGNIF TYPES
#
#####################################


class SignifLRT(Signif):
    PATTERN_PREREQ_TYPES = [PredictionLMER]
    SIGNIF_TYPE = 'lrt'
    DESCR_SHORT = 'LRT signif'
    DESCR_LONG = "Likelihood ratio significance test(s). Only available for LMER models."



