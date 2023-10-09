# hello there

#Tells what current working directory is
getwd()

dataset = read.csv('Google.csv') 

View(dataset)
str(dataset)

#summary
summary(dataset)

#Scatter Plot
with(dataset, plot(volume, close))


#variance
var(dataset$close)
#histogram 
hist(dataset$close)

#boxplot 
boxplot(close ~ symbol , data = dataset)
boxplot(high ~ symbol , data = dataset)
boxplot(low ~ symbol , data = dataset)
boxplot(open ~ symbol , data = dataset)


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

#transformation

#Normalization
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}
dataWithoutNormalization <- dataset
dataset$close<-normalize(dataWithoutNormalization$close)
dataset$volume<-normalize(dataWithoutNormalization$volume)

print(dataset)


#data before preprocessing
str(dataset)
#summary after preprocessing
summary(dataset)


#mean of closing price
mean(dataset$close)

#variance
var(dataset$close)
#histogram 
hist(dataset$close)

#boxplot 
boxplot(close ~ symbol , data = dataset)
boxplot(high ~ symbol , data = dataset)
boxplot(low ~ symbol , data = dataset)
boxplot(open ~ symbol , data = dataset)


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

#transformation

#Normalization
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}
dataWithoutNormalization <- dataset
dataset$close<-normalize(dataWithoutNormalization$close)
dataset$volume<-normalize(dataWithoutNormalization$volume)

print(dataset)

#data before preprocessing
str(dataset)
#summary after preprocessing
summary(dataset)


# load the library
library(mlbench)
library(caret)

# define the control using a random forest selection function 
# number=12 means the length of the list
control <- rfeControl(functions=rfFuncs, method="cv", number=12)
# run the RFE algorithm from column 1 to 12  
results <- rfe(dataset[,1:11],dataset[,12], sizes=c(1:11), rfeControl=control)

# summarize the results
print(results)
# list the chosen features
predictors(results)

# plot the results
plot(results, type=c("h", "o"))


# load the library        
library(mlbench)
library(caret)

# calculate correlation matrix
correlationMatrix <- cor(dataset[,3:12])

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5 )

# print indexes of highly correlated attributes

      