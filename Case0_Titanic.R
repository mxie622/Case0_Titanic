### Titanic dataset. Thanks for http://amunategui.github.io/binary-outcome-modeling/#sourcecode showing me idea of using GBM
## My work can slightly improve the model prediction precision by 8% - 10% via using different IMPUTATION METHODS. 
# Loading data
library(caret)
library(rpart)
library(pROC)
original_data <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')
titanicDF = original_data
str(titanicDF)
summary(titanicDF)
plot(titanicDF$Age)

# We notice that 'Name' is almost unique. What it contributes to modelling is it can correct the 'Age' for us. 
# EX: A Frans Olof cannot be 0.17 year-old.
# So we need to take the title out

titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',ifelse(grepl('Miss',titanicDF$Name),'Miss','Nothing'))) 
titanicDF$Title = as.factor(titanicDF$Title)
# titanicDF$Name = NULL
titanicDF <- titanicDF[c('PClass', 'Age',    'Sex',   'Title', 'Survived')]

# save the outcome for the glmnet model

tempOutcome <- titanicDF$Survived # We will compare with glmnet model that can only deal with '0' or '1'.

# Our outcome is binary: Survived or not
# titanicDF$Survived = as.factor(titanicDF$Survived)


#### There are 557 NA in 'Age' to be filled
# Step1: Take NA_set out 

###### method1: Take column by column
NA_set = titanicDF[is.na(titanicDF$Age), ]
No_NA_set = titanicDF[!is.na(titanicDF$Age), ]
# Method2: If a row has 'NA', take it out
# NA_set = titanicDF[!complete.cases(titanicDF),] ---------
# No_NA_set = titanicDF[complete.cases(titanicDF),] --------

# We use K-Nearest Neighbour to fill NA  **********
# To determine the value of k:  ***********

train(Age ~ .,
      method = "knn",
      data = No_NA_set,
      tuneGrid = expand.grid(k = c(1,3,5,7)))

# We found that k = 3 is the best option

model_to_fill_NA <- knnreg(formula = Age ~ ., data = No_NA_set, k = 3)
titanicDF$Age[is.na(titanicDF$Age)] = predict(model_to_fill_NA, subset(NA_set, select = -c(Age)))

train(Age ~ .,
      method = "knn",
      data = No_NA_set,
      tuneGrid = expand.grid(k = c(1,3,5,7)))

# Since most models only accept dummy variables
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))
print(names(titanicDF))
# [1] "PClass.1st"    "PClass.2nd"    "PClass.3rd"    "Age"           "Sex.female"    "Sex.male"     
# [7] "Title.Miss"    "Title.Mr"      "Title.Mrs"     "Title.Nothing" "Survived"     
prop.table(table(titanicDF$Survived))
# 0         1 
# 0.6572734 0.3427266 

# Split into training and testing set.
set.seed(2018)
titanicDF$Survived <- ifelse(titanicDF$Survived==1,'yes','nope')
titanicDF$Survived <- as.factor(titanicDF$Survived)
outcomeName <- 'Survived'
splitIndex <- createDataPartition(titanicDF[, "Survived"], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]
predictorsNames <- names(titanicDF)[names(titanicDF) != outcomeName]

# Models can be used

names(getModelInfo())


####### Use Gradient boosting machine   # Method 1
objControl <- trainControl(method='cv', number=5, returnResamp='none', 
                           summaryFunction = twoClassSummary, classProbs = TRUE)
objModel <- train(trainDF[, predictorsNames], as.factor(trainDF[, outcomeName]), 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))

# See what we obtain from the trained model

summary(objModel)
# var    rel.inf
# Age                     Age 38.7925808
# Sex.male           Sex.male 25.7141669
# PClass.3rd       PClass.3rd 14.8174725
# Title.Mr           Title.Mr  8.5789309
# PClass.1st       PClass.1st  6.9352029
# Title.Mrs         Title.Mrs  2.0160743
# Sex.female       Sex.female  1.0550643
# Title.Nothing Title.Nothing  0.8440612
# PClass.2nd       PClass.2nd  0.6652374
# Title.Miss       Title.Miss  0.5812088

# Make predictions

predictions <- predict(object=objModel, testDF[, predictorsNames], type='raw')
print(postResample(pred=predictions, obs=as.factor(testDF[,outcomeName])))
# Accuracy     Kappa 
# 0.8960245 0.7630637 

# Probability situation

predictions <- predict(object=objModel, testDF[,predictorsNames], type='prob')
head(predictions)
postResample(pred=predictions[[2]], obs=ifelse(testDF[,outcomeName]=='yes',1,0))
auc <- roc(ifelse(testDF[, outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)
# Area under the curve: 0.91

# We can plot the importance of each variable

plot(varImp(objModel,scale=F))

vimp <- varImp(objModel, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  



######### Use glmnet   # Method 2
# We need to convert the factor type to numeric for 'Survived'

titanicDF$Survived <- tempOutcome

# Split the whole set into training and testing sets

splitIndex <- createDataPartition(titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]

objControl_glmnet <- trainControl(method='cv', number=5, returnResamp='none')
objModel_glmnet <- train(trainDF[,predictorsNames], 
                  trainDF[, outcomeName], 
                  method='glmnet',  
                  trControl=objControl_glmnet)


# See what we obtain from the trained model

summary(objModel_glmnet)

# Make predictions

predictions_glmnet <- predict(object=objModel_glmnet, testDF[,predictorsNames])
head(predictions_glmnet)
auc <- roc(testDF[,outcomeName], predictions_glmnet)
print(auc$auc)
# Area under the curve: 0.8336 < 0.857


