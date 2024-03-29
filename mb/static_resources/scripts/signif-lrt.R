#!/usr/bin/env Rscript
options(width=200) 

#########################
#
# Loading Data and Libraries
#
#########################

args <- commandArgs(trailingOnly=TRUE)
cliargs <- args[-(1:2)]
options('warn'=1) #report non-convergences, etc

library(lme4)
library(languageR)
library(optimx)
library(ggplot2)
#The below scripts cannot be distributed with Modelblocks
# Relative path code from Suppressingfire on StackExchange
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
script.basename <- dirname(script.name)
wd = getwd()
setwd(script.basename)
source('lmer-tools.R')
setwd(wd)

basefile <- load(args[1])
base <- get(basefile)
corpus <- base$corpus

mainfile <- load(args[2])
main <- get(mainfile)


#########################
#
# Definitions
#
#########################

printSignifSummary <- function(mainName, mainEffect, basemodel, testmodel, signif, aov=NULL) {
    cat(paste0('Main effect: ', mainName), '\n')
    cat(paste0('Corpus: ', corpus), '\n')
    cat(paste0('Effect estimate (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'Estimate'], sep=''), '\n')
    cat(paste0('t value (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'t value'], sep=''), '\n')
    cat(paste0('p value: ', signif), '\n')
    cat(paste0('Baseline loglik: ', logLik(basemodel), '\n'))
    cat(paste0('Full loglik: ', logLik(testmodel), '\n'))
    cat(paste0('Log likelihood ratio: ', logLik(testmodel)-logLik(basemodel), '\n'))
    cat(paste0('Relative gradient (baseline): ',  max(abs(with(basemodel@optinfo$derivs,solve(Hessian,gradient)))), '\n'))
    convWarnBase <- is.null(basemodel@optinfo$conv$lme4$messages)
    cat(paste0('Converged (baseline): ', as.character(convWarnBase), '\n'))
    cat(paste0('Relative gradient (main effect):',  max(abs(with(testmodel@optinfo$derivs,solve(Hessian,gradient)))), '\n'))
    convWarnMain <- is.null(testmodel@optinfo$conv$lme4$messages)
    cat(paste0('Converged (main effect): ', as.character(convWarnMain), '\n'))
    if (!is.null(aov)) {
        print(aov)
    }
}

cat('Likelihood Ratio Test (LRT) Summary\n')
cat('===================================\n\n')
cat('Correlation of numeric variables in model:\n')
print(main$correlations)
cat('\n\n')
cat('SDs of numeric variables in model:\n')
for (c in names(main$sd_vals)) {
    cat(paste0(c, ': ', main$sd_vals[[c]], '\n'))
}
cat('\n')

smartPrint('Summary of baseline model')
printSummary(base$model)
smartPrint(paste('Summary of main effect (', setdiff(base$abl,main$abl), ') model', sep=''))
printSummary(main$model)

aov = anova(base$model, main$model)
signif = aov[['Pr(>Chisq)']][[2]]

cat('Significance results\n')
smartPrint(paste('Baseline vs. Main Effect (', setdiff(base$abl,main$abl), ')', sep=''))
printSignifSummary(setdiff(base$abl,main$abl),
                   setdiff(base$ablEffects,main$ablEffects),
                   base$model,
                   main$model,
                   signif,
                   aov)

if (!is.null(main$lambda)) {
    printBoxCoxInvBetas(main$beta_ms, main$lambda, main$y_mu, main$sd_vals)
}

