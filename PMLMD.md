Prediction of the manner of sport exercise activity
========================================================

Dataset pml_training contains data about correct and incorrect ways to perform sport activities. The factor variable containing information on the way the exercise was done is "classe". Using dataset pml-testing, the goal is to predict the corresponding factor value. We tried three models: TreeBag, Random Forest and Generalized Boosted Regression.

First, we load libraries and read the data 


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(gbm)
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
```

```r
pml_train <- read.csv("pml-training.csv", stringsAsFactors = F)
pml_test <- read.csv("pml-testing.csv", stringsAsFactors = F)
```

Columns 1:7 contain information which is not not relevant to the correctness of doing workout, so I removed them. Column 160 is the output "classe" for pml_train dataset and "problem_id" for pml_test. I also remove this column from data. Some of the data columns have "character" type, but contain numbers, so I converted "character" to "numeric" format. As data contains NA cells, I substituted NA with the value zero. The predicted variable is Classe = pml_train$classe.


```r
train <- pml_train[, -c(1:7, 160)]
for (i in 1:ncol(train)) {
  if (class(train[, i]) == "character") {
    suppressWarnings(train[, i] <- as.numeric(train[, i]))    
  }
}
train[is.na(train)] <- 0
Classe <- factor(pml_train[, 160])
```

The same way the pml_test is processed

```r
testset <- pml_test[, -c(1:7, 160)]
for (i in 1:ncol(testset)) {
  if (class(testset[, i]) == "character") {
    testset[, i] <- as.numeric(testset[, i])
  }
}
testset[is.na(testset)] <- 0
```

Dataset pml_train contains 19622 rows, so I chose to use only small part (p=0.3) of the data for the training, 
and the rest for the testing purposes.
Ususally ~70% of data is used for training, but when I tried to use this chunk of data, it was a problem with memory and problem with the time of running of the method, both of them were unacceptably huge.    


```r
set.seed(32323)
indexTrain <- createDataPartition(y = Classe, p = 0.3, list = FALSE)
training <- train[indexTrain, ]
testing <- train[-indexTrain, ]

trainingClasse <- Classe[indexTrain] 
testingClasse <- Classe[-indexTrain] 
```

For cross validation the number of folds is chosen not so high (5) to have low out-of-sample error and not to 
overfit the model 


```r
fitControl <- trainControl(method = "cv", number = 5)
```

Three model are tried: TreeBag, Random Forest and Generalized Boosted Regression.


```r
set.seed(32323)
fitTreeBag <- train(trainingClasse ~ ., method = "treebag", data = training, trControl = fitControl)
```

```
## Loading required package: ipred
## Loading required package: plyr
```

```r
resultTB <- predict(fitTreeBag, newdata = testing)
confusionMatrix(resultTB, testingClasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3848   55    0    1    0
##          B   35 2550   20    4   13
##          C   11   40 2350   54   12
##          D    4    9   25 2185   28
##          E    8    3    0    7 2471
## 
## Overall Statistics
##                                         
##                Accuracy : 0.976         
##                  95% CI : (0.973, 0.979)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.97          
##  Mcnemar's Test P-Value : 4.1e-12       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.985    0.960    0.981    0.971    0.979
## Specificity             0.994    0.993    0.990    0.994    0.998
## Pos Pred Value          0.986    0.973    0.953    0.971    0.993
## Neg Pred Value          0.994    0.990    0.996    0.994    0.995
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.280    0.186    0.171    0.159    0.180
## Detection Prevalence    0.284    0.191    0.180    0.164    0.181
## Balanced Accuracy       0.990    0.977    0.985    0.982    0.989
```


```r
set.seed(32323)
fitRF <- train(trainingClasse ~ ., method = "rf", data=training, prox=T, trControl = fitControl) 
resultRF <- predict(fitRF, newdata = testing)
confusionMatrix(resultRF, testingClasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3885   49    0    0    0
##          B   14 2568   19    2    5
##          C    1   30 2352   54    6
##          D    2    5   24 2190   25
##          E    4    5    0    5 2488
## 
## Overall Statistics
##                                         
##                Accuracy : 0.982         
##                  95% CI : (0.979, 0.984)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.977         
##  Mcnemar's Test P-Value : 2.27e-09      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.995    0.967    0.982    0.973    0.986
## Specificity             0.995    0.996    0.992    0.995    0.999
## Pos Pred Value          0.988    0.985    0.963    0.975    0.994
## Neg Pred Value          0.998    0.992    0.996    0.995    0.997
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.283    0.187    0.171    0.159    0.181
## Detection Prevalence    0.286    0.190    0.178    0.164    0.182
## Balanced Accuracy       0.995    0.981    0.987    0.984    0.992
```


```r
set.seed(32323)
suppressWarnings(fitGBM <- train(trainingClasse ~ ., method = "gbm", data=training, verbose=F, trControl = fitControl)) 
resultGBM <- predict(fitGBM, newdata = testing)
confusionMatrix(resultGBM, testingClasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3824  143    1    0    7
##          B   40 2403   79    6   36
##          C   13   94 2287   98   36
##          D   24    7   21 2125   35
##          E    5   10    7   22 2410
## 
## Overall Statistics
##                                         
##                Accuracy : 0.95          
##                  95% CI : (0.946, 0.954)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.937         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.979    0.904    0.955    0.944    0.955
## Specificity             0.985    0.985    0.979    0.992    0.996
## Pos Pred Value          0.962    0.937    0.905    0.961    0.982
## Neg Pred Value          0.992    0.977    0.990    0.989    0.990
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.278    0.175    0.167    0.155    0.175
## Detection Prevalence    0.289    0.187    0.184    0.161    0.179
## Balanced Accuracy       0.982    0.945    0.967    0.968    0.975
```

We can see, that the best method by the accuracy is the Random Forest (Accuracy = 98%)
We calculate the results for pml-testing dataset using this mathod


```r
testClasse <- predict(fitRF, newdata = testset)
```

## Results
In array testClasse we have the results for pml-testing dataset dataset.


```r
answers = testClasse
pml_write_files = function(x) {
  n = length(x)
  for (i in 1:n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
                col.names = FALSE)
  }
}
pml_write_files(answers)
```
