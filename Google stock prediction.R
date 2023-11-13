# hello there

#Tells what current working directory is
getwd()

dataset = read.csv('GOOG.csv') 

View(dataset)
str(dataset)

#we removed the attributes (symbol, divCash, splitFactor) as they have one value only so we do not need them
dataset=dataset[,2:12]
View(dataset)

# Convert the date column to a date format
dataset$date <- as.Date(dataset$date, format = "%Y-%m-%d %H:%M:%S")

#summary
summary(dataset)

#mean of closing price
mean(dataset$close)

#variance
var(dataset$close)

#histogram 
hist(dataset$close)

#Scatter Plot
with(dataset, plot(volume, close))

#Barplot
barplot(height = dataset$close, names.arg = dataset$date, xlab = "Date", ylab = "Closing price", main = "date vs Close")


#Data cleaning

#Checking NULL, FALSE means no null, TRUE cells means the value of the cell is null
is.na(dataset)

#find the total null values
sum(is.na(dataset))

print("Since there is no NULL values we don't need to remove any rows")

#removing close outlier

library(outliers)

OutClose = outlier(dataset$close, logical =TRUE)
sum(OutClose)
Find_outlier = which(OutClose ==TRUE, arr.ind = TRUE)
OutClose
Find_outlier

#removing volume's outlier
OutVolume = outlier(dataset$volume, logical =TRUE)
sum(OutVolume)
Find_outlier = which(OutVolume ==TRUE, arr.ind = TRUE)
OutVolume
Find_outlier

#Remove outlier
dataset= dataset[-Find_outlier,]


#Feature selection ,Remove Redundant Features
# load the library        
library(mlbench)
library(caret)
library(ggplot2)
library(lattice)

# calculate correlation matrix
correlationMatrix <- cor(dataset[,2:11])

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5 )

# print indexes of highly correlated attributes
print(highlyCorrelated)

#transformation

#data before preprocessing
View(dataset)


#Normalization
#normalization was performed to ensure consistent scaling of the data. The normalization technique applied was the max-min normalization. This technique rescales the values of specific attributes within a defined range between 0 and 1. 
#We can use the normalized dataset provides a more uniform and comparable representation of the attributes, enabling accurate analysis and modeling for stock predaction with result as shown.
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}
dataWithoutNormalization <- dataset
dataset$close<-normalize(dataWithoutNormalization$close)
dataset$volume<-normalize(dataWithoutNormalization$volume)
dataset$open<-normalize(dataWithoutNormalization$open)
dataset$low <-normalize(dataWithoutNormalization$low)
dataset$high <-normalize(dataWithoutNormalization$high)

#Discretization
#we used the Discretization technique on our class label "close" to simplify it as it has a large continuous values, we made them fall into intervals, to make it easier to analyze

dataset$close <- ifelse(dataset$close <= 0.2957251 , "low","High")
print(dataset)

#We encoded close data into factors, which would help the model read this data easily
dataset$close <- factor(dataset$close,levels = c("low", "High"), labels = c("1", "2"))

#summary after preprocessing
print(dataset)
View(dataset)
summary(dataset)


#Feature selection ,Feature selection using Recursive Feature Elimination or RFE
# load the library
library(mlbench)
library(caret)

# define the control using a random forest selection function 
# number=12 means the length of the list
control <- rfeControl(functions=rfFuncs, method="cv", number=11)
# run the RFE algorithm from column 1 to 11  
results <- rfe(dataset[,1:10],dataset[,11], sizes=c(1:10), rfeControl=control)

# summarize the results
print(results)
# list the chosen features
predictors(results)

# plot the results
plot(results, type=c("h", "o"))

#Classification:

#We will choose the attributes with the highest importance (from feature selection) to create a tree:

# a fixed random seed to make results reproducible
set.seed(1234)

# 1.Split the datasets into two subsets: Training(70%) and Testing(30%):
ind1 <- sample(2, nrow(dataset), replace=TRUE, prob=c(0.70 , 0.30))
trainData  <- dataset[ind1==1,]
testData <- dataset[ind1==2,]

# 2.Determine the predictor attributes and the class label attribute.( the formula):

library(party)    
#myFormula 
myFormula <- close ~high+low+open

# 3.Build a decision tree using training set and check the Prediction:
dataset_ctree <- ctree(myFormula, data=trainData)
table(predict(dataset_ctree), trainData$close)

# 4.Print and plot the tree:

print(dataset_ctree)
plot(dataset_ctree, type="simple")

# 5.Use the constructed model to predict the class labels of test data:
testPred <- predict(dataset_ctree, newdata = testData)
result<-table(testPred, testData$close)
result


# Evaluate the model and create confusion matrix
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
library(e1071)
library(caret)

co_result <- confusionMatrix(result)

print(co_result)
sensitivity(as.table(co_result))
specificity(as.table(co_result))

acc <- co_result$overall["Accuracy"]
acc


#2. Building the Tree using Gini Index(CART)

# For decision tree model
install.packages("rpart")
library(rpart)
# For data visualization
library(rpart.plot)

dataset.cart <- rpart(myFormula, data = trainData, method = "class", parms = list(split = "gini"))


# Visualizing the unpruned tree

library(rpart.plot)
rpart.plot(dataset.cart)


# Checking the order of variable importance

dataset.cart$variable.importance
pred.tree = predict(dataset.cart, testData, type = "class")

table(pred.tree,testData$close)

# 5.Use the constructed model to predict the class labels of test data:
testPred <- predict(dataset_ctree, newdata = testData)
result<-table(testPred, testData$close)
result

# Evaluate the model and create confusion matrix
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
library(e1071)
library(caret)

co_result <- confusionMatrix(result)

print(co_result)
sensitivity(as.table(co_result))
specificity(as.table(co_result))

acc <- co_result$overall["Accuracy"]
acc

#3. Building the Tree using Gain ratio(C5)

install.packages("caret")
install.packages("C50")
install.packages("printr")

library(C50)
library(printr)
library(caret)
#train using the trainData and create the c5.0 gain ratio tree
CloseTree <- C5.0(myFormula, data=trainData)
summary(CloseTree)
plot(CloseTree)

# a fixed random seed to make results reproducible
set.seed(1234)

# 1.Split the datasets into two subsets: Training(60%) and Testing(40%):
ind1 <- sample(2, nrow(dataset), replace=TRUE, prob=c(0.60 , 0.40))
trainData  <- dataset[ind1==1,]
testData <- dataset[ind1==2,]

# 2.Determine the predictor attributes and the class label attribute.( the formula):

library(party)    
#myFormula 
myFormula <- close ~high+low+open

# 3.Build a decision tree using training set and check the Prediction:
dataset_ctree <- ctree(myFormula, data=trainData)
table(predict(dataset_ctree), trainData$close)

# 4.Print and plot the tree:

print(dataset_ctree)
plot(dataset_ctree, type="simple")

# 5.Use the constructed model to predict the class labels of test data:
testPred <- predict(dataset_ctree, newdata = testData)
result<-table(testPred, testData$close)
result


# Evaluate the model and create confusion matrix
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
library(e1071)
library(caret)

co_result <- confusionMatrix(result)

print(co_result)
sensitivity(as.table(co_result))
specificity(as.table(co_result))

acc <- co_result$overall["Accuracy"]
acc


#2. Building the Tree using Gini Index(CART)

# For decision tree model
install.packages("rpart")
library(rpart)
# For data visualization
library(rpart.plot)

dataset.cart <- rpart(myFormula, data = trainData, method = "class", parms = list(split = "gini"))


# Visualizing the unpruned tree

library(rpart.plot)
rpart.plot(dataset.cart)


# Checking the order of variable importance

dataset.cart$variable.importance
pred.tree = predict(dataset.cart, testData, type = "class")

table(pred.tree,testData$close)

# 5.Use the constructed model to predict the class labels of test data:
testPred <- predict(dataset_ctree, newdata = testData)
result<-table(testPred, testData$close)
result

# Evaluate the model and create confusion matrix
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
library(e1071)
library(caret)

co_result <- confusionMatrix(result)

print(co_result)
sensitivity(as.table(co_result))
specificity(as.table(co_result))

acc <- co_result$overall["Accuracy"]
acc

#3. Building the Tree using Gain ratio(C5)

install.packages("caret")
install.packages("C50")
install.packages("printr")

library(C50)
library(printr)
library(caret)
#train using the trainData and create the c5.0 gain ratio tree
CloseTree <- C5.0(myFormula, data=trainData)
summary(CloseTree)
plot(CloseTree)

# a fixed random seed to make results reproducible
set.seed(1234)

# 1.Split the datasets into two subsets: Training(80%) and Testing(20%):
ind1 <- sample(2, nrow(dataset), replace=TRUE, prob=c(0.8 , 0.2))
trainData  <- dataset[ind1==1,]
testData <- dataset[ind1==2,]

# 2.Determine the predictor attributes and the class label attribute.( the formula):

library(party)    
#myFormula 
myFormula <- close ~high+low+open

# 3.Build a decision tree using training set and check the Prediction:
dataset_ctree <- ctree(myFormula, data=trainData)
table(predict(dataset_ctree), trainData$close)

# 4.Print and plot the tree:

print(dataset_ctree)
plot(dataset_ctree, type="simple")

# 5.Use the constructed model to predict the class labels of test data:
testPred <- predict(dataset_ctree, newdata = testData)
result<-table(testPred, testData$close)
result


# Evaluate the model and create confusion matrix
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
library(e1071)
library(caret)

co_result <- confusionMatrix(result)

print(co_result)
sensitivity(as.table(co_result))
specificity(as.table(co_result))

acc <- co_result$overall["Accuracy"]
acc


#2. Building the Tree using Gini Index(CART)

# For decision tree model
install.packages("rpart")
library(rpart)
# For data visualization
library(rpart.plot)

dataset.cart <- rpart(myFormula, data = trainData, method = "class", parms = list(split = "gini"))


# Visualizing the unpruned tree

library(rpart.plot)
rpart.plot(dataset.cart)


# Checking the order of variable importance

dataset.cart$variable.importance
pred.tree = predict(dataset.cart, testData, type = "class")

table(pred.tree,testData$close)

# 5.Use the constructed model to predict the class labels of test data:
testPred <- predict(dataset_ctree, newdata = testData)
result<-table(testPred, testData$close)
result

# Evaluate the model and create confusion matrix
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
library(e1071)
library(caret)

co_result <- confusionMatrix(result)

print(co_result)
sensitivity(as.table(co_result))
specificity(as.table(co_result))

acc <- co_result$overall["Accuracy"]
acc

#3. Building the Tree using Gain ratio(C5)

install.packages("caret")
install.packages("C50")
install.packages("printr")

library(C50)
library(printr)
library(caret)
#train using the trainData and create the c5.0 gain ratio tree
CloseTree <- C5.0(myFormula, data=trainData)
summary(CloseTree)
plot(CloseTree)

#Clustring:

# prepreocessing 
#Data types should be transformed into numeric types before clustering.
dataset<-dataset[,3:11]
dataset <- scale(dataset)
View(dataset)

# Load necessary packages
#install.packages("caret")
#install.packages("cluster")
#install.packages("fpc")
#install.packages("ggplot2")
library(caret)
library(cluster)
library(fpc)
library(ggplot2)

# Apply k-means clustering for different values of K

# k-means clustering to find 4 clusters 
#set a seed for random number generation  to make the results reproducible
set.seed(8953)
kmeans.result <- kmeans(dataset, 4)

# print the clustering result
print(kmeans.result)

# visualize clustering
#install.packages("factoextra")
library(factoextra)
fviz_cluster(kmeans.result, data = dataset)

# plot cluster points
plot(dataset[, c("close","open")], col = (kmeans.result$cluster) )
# plot cluster centers
points(kmeans.result$centers[, c("close","open")],  col = 1:4, pch = 8, cex=2)

# run k-means clustering to find 3 clusters
#set a seed for random number generation  to make the results reproducible
set.seed(8953)
kmeans.result <- kmeans(dataset, 3)

# print the clustering result
print(kmeans.result)

#4- visualize clustering
#install.packages("factoextra")
library(factoextra)
fviz_cluster(kmeans.result, data = dataset)

# run k-means clustering to find 2 clusters
#set a seed for random number generation  to make the results reproducible
set.seed(8953)
kmeans.result <- kmeans(dataset, 2)

# print the clustering result
print(kmeans.result)

#4- visualize clustering
#install.packages("factoextra")
library(factoextra)
fviz_cluster(kmeans.result, data = dataset)


install.packages("fpc")
library(fpc)
#kmeansruns() : It calls  kmeans() to perform  k-means clustering
#It initializes the k-means algorithm several times with random points from the data set as means.
#It estimates the number of clusters by index or average silhouette width
kmeansruns.result <- kmeansruns(dataset)  
kmeansruns.result
fviz_cluster(kmeansruns.result, data = dataset)

#k-mediods clustering with PAM
#install.packages("cluster")
library(cluster)
# group into 4 clusters
pam.result <- pam(dataset, 4)
plot(pam.result)

#Hierarchical Clustering
##----Hierarchical Clustering of the Data-----##
set.seed(2835)
# draw a sample of 40 records from the dataset data, so that the clustering plot will not be over crowded
idx <- sample(1:dim(dataset)[1], 40)
dataset2 <- dataset[idx, ]
## hiercrchical clustering
library(factoextra) 
hc.cut <- hcut(dataset2, k = 2, hc_method = "complete") # Computes Hierarchical Clustering and Cut the Tree


# Visualize dendrogram
fviz_dend(hc.cut,rect = TRUE)  #logical value specifying whether to add a rectangle around groups.
# Visualize cluster
fviz_cluster(hc.cut, ellipse.type = "convex") # Character specifying frame type. Possible values are 'convex', 'confidence' etc

#Validation

#average silhouette for each clusters 
library(cluster)
avg_sil <- silhouette(kmeans.result$cluster,dist(dataset)) #a dissimilarity object inheriting from class dist or coercible to one. If not specified, dmatrix must be.
fviz_silhouette(avg_sil)#k-means clustering with estimating k and initializations


#  define function to compute average silhouette for k clusters using silhouette()
silhouette_score <- function(k){ 
  km <- kmeans(USArrests, centers = k,nstart=25) # if centers is a number, how many random sets should be chosen?
  ss <- silhouette(km$cluster, dist(USArrests))
  sil<- mean(ss[, 3])
  return(sil)
}

# k cluster range from 2 to 10
k <- 2:10
##  call  function fore k value
avg_sil <- sapply(k, silhouette_score)  ##Apply a Function over a List or Vector
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)


# silhouette method
#install.packages("NbClust")
library(NbClust)
#a)fviz_nbclust() with silhouette method using library(factoextra) 
fviz_nbclust(dataset, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
#b) NbClust validation
fres.nbclust <- NbClust(dataset, distance="euclidean", min.nc = 2, max.nc = 10, method="kmeans", index="all")

# Elbow method for determining the optimal number of clusters (k-means)
wss <- numeric(length = 10)
for (k in 1:10) {
  kmeans_model <- kmeans(dataset, centers = k, nstart = 10)
  wss[k] <- sum(kmeans_model$close)
  
 
  # Evaluate BCubed precision and recall for k-medoids
  bcubed_kmedoids <- cluster.stats(dissimilarity_matrix, kmedoids_model$cluster)$bcubed
  cat("\nBCubed Precision and Recall for K-Medoids Clustering:\n")
  print(bcubed_kmedoids)
  
  # Extract the total within-cluster sum of squares (TWSS)
  twss <- sum(kmeans.result$withinss)
  # Print the TWSS
  cat(paste("Total Within-Cluster Sum of Squares (TWSS):", twss, "\n"))
  
  # Install and load required libraries
  library(caret)
  library(ggplot2)
  library(lattice)
  
  
  # Assuming you have true labels and predicted labels
  true_labels <- c(1, 1, 1, 0, 0, 1, 0, 1, 0, 1)
  predicted_labels <- c(1, 0, 1, 0, 0, 1, 0, 1, 1, 1)
  
  # Create a confusion matrix
  conf_matrix <- confusionMatrix(factor(predicted_labels), factor(true_labels))
  
  # Extract recall from the confusion matrix
  recall <- conf_matrix$byClass["Sensitivity"]
  
  # Print the result
  cat(paste("Recall:", recall, "\n"))
}