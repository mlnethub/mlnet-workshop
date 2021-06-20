#Ted Kwartler
#Ted@sportsanalytics.org
#St Peters Workshop: Intro to Text Mining using R
#11-2-2015
#v3.0 Basics: Cleaning and Frequency Count

#Set the working directory
#setwd('/Users/ted/Desktop/St Peters')

#libraries
library(tm)

#options, functions
options(stringsAsFactors = FALSE) #text strings will not be factors of categories
Sys.setlocale('LC_ALL','C') #some tweets are in different languages so you may get an error

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
custom.stopwords <- c(stopwords('english'), 'lol', 'smh')

#bring in some text
text<-read.csv('coffee.csv', header=TRUE)

#Keep the meta data, apply the functions to make a clean corpus
dd<-data.frame(id=text$id,text=text$text)
custom.reader <- readTabular(mapping=list(content="text", id="id"))
corpus <- VCorpus(DataframeSource(dd), readerControl=list(reader=custom.reader))
corpus<-clean.corpus(corpus)

#Make a Document Term Matrix or Term Document Matrix depending on analysis
dtm<-DocumentTermMatrix(corpus)
tdm<-TermDocumentMatrix(corpus)
dtm.tweets.m<-as.matrix(dtm)
tdm.tweets.m<-as.matrix(tdm)

dtm.tweets.m[1:5,100:110]
tdm.tweets.m[100:110,1:5]

#End