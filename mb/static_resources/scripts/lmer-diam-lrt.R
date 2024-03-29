#!/usr/bin/env Rscript
options(width=200) 

#########################
#
# Loading Data and Libraries
#
#########################

library(optparse)
opt_list <- list(
    make_option(c('-t', '--title'), type='character', default='title', help='Title of "diamond" ANOVA (e.g. "X vs. Y").'),
    make_option(c('-l', '--left'), type='character', default='effect1', help='Name of "left" model (effect 1).'),
    make_option(c('-r', '--right'), type='character', default='effect2', help='Name of "right" model (effect 2).')
)

library(lme4)
library(languageR)

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

opt_parser <- OptionParser(option_list=opt_list)
opts <- parse_args(opt_parser, positional_arguments=4)
params <- opts$options

basefile <- load(opts$args[1])
base <- get(basefile)
corpus <- base$corpus

leftfile <- load(opts$args[2])
left <- get(leftfile)

rightfile <- load(opts$args[3])
right <- get(rightfile)

bothfile <- load(opts$args[4])
both <- get(bothfile)


#########################
#
# Definitions
#
#########################

printSignifSummary <- function(mainName, mainEffect, basemodel, testmodel, signif) {
    cat(paste0('Main effect: ', mainName), '\n')
    cat(paste0('Corpus: ', corpus), '\n')
    cat(paste0('Effect estimate (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'Estimate'], sep=''), '\n')
    cat(paste0('t value (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'t value'], sep=''), '\n')
    cat(paste0('Significance (Pr(>Chisq)): ', signif[['Pr(>Chisq)']][[2]]), '\n')
    cat(paste0('Relative gradient (baseline): ',  max(abs(with(basemodel@optinfo$derivs,solve(Hessian,gradient)))), '\n'))
    convWarnBase <- is.null(basemodel@optinfo$conv$lme4$messages)
    cat(paste0('Converged (baseline): ', as.character(convWarnBase), '\n'))
    cat(paste0('Relative gradient (main effect):',  max(abs(with(testmodel@optinfo$derivs,solve(Hessian,gradient)))), '\n'))
    convWarnMain <- is.null(testmodel@optinfo$conv$lme4$messages)
    cat(paste0('Converged (main effect): ', as.character(convWarnMain), '\n'))
    print(signif)
}

cat('Diamond Likelihood Ratio Test (LRT) Summary\n')
cat('===========================================\n\n')
cat('Correlation of numeric variables in model:\n')
print(both$correlations)
cat('\n\n')
cat('SDs of numeric variables in model:\n')
for (c in names(both$sd_vals)) {
    cat(c, ': ', both$sd_vals[[c]], '\n')
}
cat('\n')

smartPrint('Base summary:')
printSummary(base$model)
smartPrint('Left summary:')
printSummary(left$model)
smartPrint('Right summary:')
printSummary(right$model)
smartPrint('Both summary:')
printSummary(both$model)

smartPrint(paste('Diamond Anova:',params$title))
smartPrint(paste('Effect 1 (', params$left, ') vs. Baseline', sep=''))
printSignifSummary(setdiff(base$abl,left$abl), 
                   setdiff(base$ablEffects,left$ablEffects),
                   base$model,
                   left$model,
                   anova(base$model, left$model))
if (!is.null(left$lambda)) {
    printBoxCoxInvBetas(left$beta_ms, left$lambda, left$y_mu, left$sd_vals)
}

smartPrint(paste('Effect 2 (', params$right, ') vs. Baseline', sep=''))
printSignifSummary(setdiff(base$abl,right$abl), 
                   setdiff(base$ablEffects,right$ablEffects),
                   base$model,
                   right$model,
                   anova(base$model, right$model))
if (!is.null(right$lambda)) {
    printBoxCoxInvBetas(right$beta_ms, right$lambda, right$y_mu, right$sd_vals)
}

smartPrint(paste('Both vs. Effect 1 (', params$left, '):', sep=''))
printSignifSummary(setdiff(left$abl,both$abl), 
                   setdiff(left$ablEffects,both$ablEffects),
                   left$model,
                   both$model,
                   anova(left$model, both$model))
if (!is.null(both$lambda)) {
    printBoxCoxInvBetas(both$beta_ms, both$lambda, both$y_mu, both$sd_vals)
}

smartPrint(paste('Both vs. Effect 2 (', params$right, '):', sep=''))
printSignifSummary(setdiff(right$abl,both$abl), 
                   setdiff(right$ablEffects,both$ablEffects),
                   right$model,
                   both$model,
                   anova(right$model, both$model))
if (!is.null(both$lambda)) {
    printBoxCoxInvBetas(both$beta_ms, both$lambda, both$y_mu, both$sd_vals)
}

