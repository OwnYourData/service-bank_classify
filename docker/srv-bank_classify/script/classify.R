# cat input.json | Rscript classify.R

library(methods)
library(jsonlite)
library(NLP)
library(tm)
library(naivebayes)

# get data from STDIN ====
myStdin <- file("stdin")
input <- suppressWarnings(readLines(myStdin))
close(myStdin)
inputParsed <- fromJSON(input)

# validation ====
if('bookings' %in% names(inputParsed)){
        bookings_raw <- inputParsed$bookings 
} else {
        stop('ungültiges Format: Array "bookings" fehlt')
}

if('tagged' %in% names(inputParsed)){
        tagged_raw <- inputParsed$tagged 
} else {
        stop('ungültiges Format: Array "tagged" fehlt')
}

# text mining ====
clean_corpus <- function(corpus){
        corpus <- tm_map(corpus, stripWhitespace)
        corpus <- tm_map(corpus, removePunctuation)
        corpus <- tm_map(corpus, content_transformer(tolower))
        # corpus <- tm_map(corpus, removeWords, c(stopwords("de")))
        return(corpus)
}

# raw bookings
bookings_matrix <- as.matrix(
        DocumentTermMatrix(
                clean_corpus(
                        VCorpus(
                                VectorSource(bookings_raw$other$original)
                        )
                )
        )
)
bookings <- matrix(as.factor(bookings_matrix), 
                   nrow=nrow(bookings_matrix), 
                   ncol=ncol(bookings_matrix))
colnames(bookings) <- colnames(bookings_matrix)

# tagged bookgings ====
tagged_matrix <- as.matrix(
        DocumentTermMatrix(
                clean_corpus(
                        VCorpus(
                                VectorSource(tagged_raw$other$original)
                        )
                )
        )
)
tagged <- matrix(as.factor(tagged_matrix),
                 nrow=nrow(tagged_matrix),
                 ncol=ncol(tagged_matrix))
colnames(tagged) <- colnames(tagged_matrix)
predictor <- tagged_raw$type

# naive bayes for classification ====
model <- naive_bayes(tagged,  as.factor(predictor))
bookings_raw$type <- as.character(predict(model, bookings, type='class'))

# write output ====
toJSON(bookings_raw, pretty = TRUE, auto_unbox = TRUE)
