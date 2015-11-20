# initialize environment
install.packages("tm")
install.packages("plyr")
install.packages("class")
install.packages("RCurl")
install.packages("caTools")
install.packages("rminer")

libs <- c("tm","plyr","class","RCurl","caTools","rminer")
lapply(libs, require, character.only = TRUE)
options(stringsAsFactors = FALSE)

#read the taining data set
data <- read.csv(" train.csv", header=TRUE)

#remove unused columns
data$ID<-NULL

#the insult column is converted into a factor
data$Insult <- as.factor(data$Insult)

#function to clear the noice from the comments
cleanCorpus<- function(corpus)
{
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  return (corpus)
}

#combine the comment of both the test and train corpus
corpusConvert<- Corpus(VectorSource(data$Comment))

#get the term document matrix with frequency counts by first cleaning the corpus with all the sentences
tdmInsNIns <- DocumentTermMatrix(cleanCorpus(corpusConvert))

#remove all the sparse terms from the term document matrix
removeSparse <- removeSparseTerms(tdmInsNIns, 0.99)

#once the sparse terms are removed, we are left with critical words
criticalWords <- as.data.frame(as.matrix(removeSparse))
colnames(criticalWords) <- make.names(colnames(criticalWords))

#get the train words from the critical words and the test words from critical words
criticalTrainWords <- head(criticalWords, nrow(data))

# Next we need to classifiy each row as either insult or not insult, this is done by binding this with original
#data and then removing the comment column, this leaves the insult classification for each word with their frequencies
wordsFreqWithInsultType <- cbind(Insult =data$Insult, criticalTrainWords)

#get the sample that we are going to use for training the model, the rest is used for testing
#here we use 85% as the trianing sample
trainSample <- sample.split(wordsFreqWithInsultType$Insult, .85)

# WHERE sample is true, the set belongs to 85% of the training sample and hence considered as training data
#rest are considered test data
trainSet <- wordsFreqWithInsultType[trainSample==T,]
testSet <- wordsFreqWithInsultType[trainSample!=T,]

#used for getting reproducible random results
set.seed(1234)

#we are going to create 4 graphs and output them in 2 rows and 2 columns
old.par <- par(mfrow=c(2,2))

#test dataset for combining with predicted result
train <- sample.split(data$Insult, .85)
test <- data[train!=T,]

#modeling using naiveBayes
M<-fit(Insult~.,data=trainSet,model="naiveBayes",task="class")
summary(M)

#predict the test dataset using the model
P=predict(M,testSet)
summary(P)

#find the accuracy of the prediction 
acc<-print(mmetric(testSet$Insult,P,"ACC"))

#compute the confusion matrix for plotting
C=mmetric(testSet$Insult,P,metric="CONF")
print(C$conf)

#find the matrix to plot it in a graph
z = matrix(C$conf, ncol=2)

#name the rows and column of the graph
colnames(z) = c("Not Insult","Insult")
rownames(z) = c("Not Insult","Insult")

#plotting confusion matrix into graph
ctable <- as.table(z)
naiveBayes<-fourfoldplot(ctable, color = c("#CC6866", "#99dC99"),
             conf.level = 0, margin = 1, main = paste("Naive Bayes Confusion Matrix - Acc:",as.String(acc),sep=""))

#saving output in the directory
write.csv(P, file = "NaiveBayesOutput.csv")
res <- read.csv("NaiveBayesOutput.csv")
output<- cbind(res$x,test$Comment)
colnames(output) = c("Insult","Comment")
write.csv(output,"NaiveBayesOutput.csv")

#modeling using Decision Tree
M<-fit(Insult~.,data=trainSet,model="dt",task="class")
summary(M)

#predict the test dataset using the model
P=predict(M,testSet)
summary(P)

#find the accuracy of the prediction 
acc<-print(mmetric(testSet$Insult,P,"ACC"))

#compute the confusion matrix for plotting
C=mmetric(testSet$Insult,P,metric="CONF")
print(C$conf)

#find the matrix to plot it in a graph
z = matrix(C$conf, ncol=2)

#name the rows and column of the graph
colnames(z) = c("Not Insult","Insult")
rownames(z) = c("Not Insult","Insult")

#plotting confusion matrix into graph
ctable <- as.table(z)
dt<-fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = paste("Decision Tree Confusion Matrix - Acc:",as.String(acc),sep=""))

#saving output in the directory
write.csv(P, file = "DecisionTreeOutput.csv")
res <- read.csv("DecisionTreeOutput.csv")
output<- cbind(res$x,test$Comment)
colnames(output) = c("Insult","Comment")
write.csv(output,"DecisionTreeOutput.csv")

#modeling using logistic regression
M<-fit(Insult~.,data=trainSet,model="lr",task="class")
summary(M)

#predict the test dataset using the model
P=predict(M,testSet)
summary(P)

#find the accuracy of the prediction 
acc<-print(mmetric(testSet$Insult,P,"ACC"))

#compute the confusion matrix for plotting
C=mmetric(testSet$Insult,P,metric="CONF")
print(C$conf)

#find the matrix to plot it in a graph
z = matrix(C$conf, ncol=2)

#name the rows and column of the graph
colnames(z) = c("Not Insult","Insult")
rownames(z) = c("Not Insult","Insult")

#plotting confusion matrix into graph
ctable <- as.table(z)
lr<-fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main =  paste("Logistic Regression Confusion Matrix - Acc:",as.String(acc),sep=""))

#saving output in the directory
write.csv(P, file = "LogisticRegressionOutput.csv")
res <- read.csv("LogisticRegressionOutput.csv")
output<- cbind(res$x,test$Comment)
colnames(output) = c("Insult","Comment")
write.csv(output,"LogisticRegressionOutput.csv")

#modeling using support vector machine 
M<-fit(Insult~.,data=trainSet,model="ksvm",task="class")
summary(M)

#predict the test dataset using the model
P=predict(M,testSet)
summary(P)

#find the accuracy of the prediction 
acc<-print(mmetric(testSet$Insult,P,"ACC"))

#compute the confusion matrix for plotting
C=mmetric(testSet$Insult,P,metric="CONF")
print(C$conf)

#find the matrix to plot it in a graph
z = matrix(C$conf, ncol=2)

#name the rows and column of the graph
colnames(z) = c("Not Insult","Insult")
rownames(z) = c("Not Insult","Insult")

#plotting confusion matrix into graph
ctable <- as.table(z)
svm<-fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = paste("SVM Confusion Matrix - Acc:",as.String(acc),sep=""))

#saving output in the directory
write.csv(P, file = "SVMOutput.csv")
res <- read.csv("SVMOutput.csv")
output<- cbind(res$x,test$Comment)
colnames(output) = c("Insult","Comment")
write.csv(output,"SVMOutput.csv")
#put the graphs in 2 row 2 column format
par(old.par)

