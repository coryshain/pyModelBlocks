#!/usr/bin/env Rscript
library(plyr)
library(dplyr)
library(ggplot2)

#read in RT data from 2 separate files
b2 <- read.csv('batch2_pro.csv')
b1 <- read.csv('batch1_pro.csv')
d <- rbind(b1, b2)

##subtract 2 from zone to properly align region...should confirm with Hal that this is correct,
## but the RTs seem to line up correctly in plots
d$zone <- d$zone - 2

#read in story words and region
#item is story (1-10), zone is RT region
word.df <- read.csv('all_stories.tok', sep = '\t')
d <- merge(d, word.df, by= c('item', 'zone'), all.x = T, all.y = T)

#remove regions that do not have words
d <- filter(d, !is.na(word))

#exclude stories where subject does not get more than 4/6 correct
# unfiltered <- d
# d <- filter(d, correct > 4)

#exclude data points less than 50 ms, greater than 3000 ms
# d <- d[d$RT > 100 & d$RT < 3000, ]
# d$l <- nchar(as.character(d$word))


#calculate by-word statistics

gmean <- function(x) exp(mean(log(x)))
gsd   <- function(x) exp(sd(log(x)))

word.info = d %>%
  group_by(word, zone, item) %>%
    summarise(nItem=length(RT),
              meanItemRT=mean(RT),
	      sdItemRT=sd(RT),
	      gmeanItemRT=sd(RT),
	      gsdItemRT=gsd(RT))

d <- inner_join(d, word.info, by=c("word", "zone", "item"))

# write processed output, by word, overall
write.table(word.info, 'processed_wordinfo_MB.tsv', quote = F, row.names=F, sep="\t")
write.table(d, 'processed_RTs_MB.tsv', quote=F, row.names=F, sep="\t")
