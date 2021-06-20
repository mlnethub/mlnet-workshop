#Ted Kwartler
#Ted@sportsanalytics.org
#St Peters Workshop: Intro to Text Mining using R
#11-2-2015
#v7.0 Topic Modeling and simple sentiment/polarity

#Set the working directory
setwd('/Users/ted/Desktop/St Peters')

#libraries
library(qdap)
library(tm)
library(topicmodels)
library(portfolio) 

#options, functions
options(stringsAsFactors = FALSE)
Sys.setlocale('LC_ALL','C')

#try to lower function
tryTolower <- function(x){
  # return NA when there is an error
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error = function(e) e)
  # if not an error
  if (!inherits(try_error, 'error'))
    y = tolower(x)
  return(y)
}

clean.corpus<-function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(tryTolower))
  corpus <- tm_map(corpus, removeWords, custom.stopwords)
  return(corpus)
}

#Create custom stop words
custom.stopwords <- c(stopwords('english'), 'lol', 'smh', 'amp', 'chardonnay')

#Bigram token maker
bigram.tokenizer <-function(x)
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

#Bring in data
text<-read.csv(file="chardonnay2.csv", head=TRUE)

#Create a clean corpus
corpus <- VCorpus(DataframeSource(data.frame(text$text)))
corpus <-clean.corpus(corpus)

#Make a DTM
dtm<-DocumentTermMatrix(corpus, control=list(tokenize=bigram.tokenizer))


#In Topic Modeling, remove any docs with all zeros after removing stopwords
rowTotals <- apply(dtm , 1, sum) 
dtm.new   <- dtm[rowTotals> 0, ]

#In Sentiment, to ensure the number of rows in the dtm.new and the sentiment data frame equal
text <-cbind(text,rowTotals)
text <- text[rowTotals> 0, ]

#Begin Topic Modeling; can use CTM or LDA
topic.model <- CTM(dtm.new, k = 5) 

#Topic Extraction
topics<-get_terms(topic.model, 5)
colnames(topics)<-c("topic1","topic2","topic3","topic4","topic5")
topics<-as.data.frame(topics)
t1<-paste(topics$topic1,collapse=' ') 
t2<-paste(topics$topic2,collapse=' ') 
t3<-paste(topics$topic3,collapse=' ') 
t4<-paste(topics$topic4,collapse=' ') 
t5<-paste(topics$topic5,collapse=' ') 

#Score each tweet's probability for the topic models then add the topic words to the df as headers
scoring<-posterior(topic.model)
scores<-scoring$topics
scores<-as.data.frame(scores)
colnames(scores)<-c(t1,t2,t3,t4,t5)

#The max probability of each tweet classifies the tweet document
topics.text<-as.data.frame(cbind(row.names(scores),apply(scores,1,function(x) names(scores)[which(x==max(x))]))) 

#Apply the subjective lexicon scoring function
#library(qdap)
sentiment.score<-polarity(text$text)#, pos,neg, .progress='text')

#Get the length of each doc by number of words not characters
doc.length<-rowSums(as.matrix(dtm.new))

#Create a unified data frame
all<-cbind(topics.text,scores,sentiment.score[[1]][3], doc.length)
names(all)[2]<-paste("topic")
names(all)[8]<-paste("sentiment")
names(all)[9]<-paste("length")
all[all == ""] <- NA

#Make the treemap
map.market(id=all$V1, area=all$length, group=all$topic, color=all$sentiment, main="Sentiment/Color, Length/Area, Group/Topic")

#End