# Practical ML - Weight Lifting Assignment
Jeroen van Meijgaard  
7/14/2017  

## Machine Learning Assignment - Approach

For the purposes of this exercise we were provided with a data set of individuals wearing accelerometers on various places on their body while performing a specific weight lifting exercise. The exercise was monitored by qualified trainers, and was performed correctly or incorrectly in various ways. The outcome variable was categorized in 5 categories (A, B, C, D, or E). The measurement variables are provided for various points in time during each individual repetition, and summary variables are provided for each time window (I am assuming that a time window represents a single repetition that is either done correctly or incorrectly). The measurement variables provide specific data elements from the accelerometers.

Since the test data reflects observations of a single time point during the exercise, no summary data is available for the each exercise. Thus there is no point in using the summary data in the machine learning algorithm as it will not help in predicting the outcomes in the test set (and presumably in other new data provided)

First I will obtain the data from the provided location and store locally. Next, read in the data and then split the training sample into training and validating set (80/20 random split). All the summary data elements are dropped as well as time stamps and time value in both training and validating sets.

I will create two models using different machine learning algorithms: gradient boosting and random forest. As it turns out the results on the validating set show very high accuracy (over 99%) for each algorithm, and a very high cross match between to two algorithms, so no further optimization was conducted, e.g. using stacking, as further gains will likely be minimal. Moreover, each algorithm yields the same predictions on the test set.

Each of the ML algorithms required some tuning to ensure high accuracy, and limit the amount of computing time required. For the random forest algorithm I chose to use tunelength=30 and sample=’random’ which provided high accuracy within a reasonable time. For the gradient boosting algorithm I used a grid search using tuneGrid, and set n.trees up to 500 and an interaction.depth up to 7.





## Get data and save locally

Check if previously downloaded; download and unzip the files.

Read in the files as pml_training and pml_testing.

All numeric data fields in pml_training are then converted to numeric.


```r
setwd('~/Analytics/coursera')

#initialize the url and the zip filename
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fname <- "data/pml-training.csv"

## Download and unzip the dataset:
if (!file.exists("data/pml-training.csv")){ 
    download.file(url, fname)
}

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
fname <- "data/pml-testing.csv"

## Download and unzip the dataset:
if (!file.exists("data/pml-training.csv")){ 
    download.file(url, fname)
}

pml_training <- read.csv('data/pml-training.csv',colClasses='character')

pml_testing <- read.csv('data/pml-testing.csv')
```

## Split training sample

Split training sample into 80% training sample and 20% validation sample

Then we need to drop a number of features that would not realistically be available. For example the time window variable perfectly predicts the classe variable (and so would do it for the test sample as well)

Also, the data set contains summary data for each time window. However, form the testing sample it looks like it's assumed that prediction needs to happen based on a single time point. We will drop all summary variables, and retain all variables that are avaialble for each point in time. 

Additionally we will keep the name of the participant (to allow for an individual effect).



```r
#split training into train and validation sets (80/20 split)
set.seed(61253)
inTrain <- createDataPartition(pml_training$classe, p = 0.80, list = FALSE)

training <- pml_training[inTrain,]
validating <- pml_training[-inTrain,]

#select only variables with values at every observation (i.e. not the summary variables)
n<-names(training)
vars <- c(grep('^gyros',n),grep('^accel',n),grep('^magnet',n),grep('^roll',n),grep('^pitch',n),grep('^yaw',n),grep('^total',n))

#also take the 'classe' variable and 'user_name' which may provide an individual effect component
training_sub <- data.frame(training[,c('classe','user_name')],lapply(training[,n[vars]],as.numeric))

#create subset from validating accordingly
validating_sub <- data.frame(validating[,c('classe','user_name')],lapply(validating[,n[vars]],as.numeric))
```
## Train a model - Gradient Boosting



```r
set.seed(11894)
gbmGrid <-  expand.grid(interaction.depth = c(3, 5, 7), 
                        n.trees = (1:5)*100, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

modelFit_gbm <- train(classe~., method='gbm', data=training_sub,
                      trControl=trainControl(method='cv', number=5),tuneGrid = gbmGrid)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2331
##      2        1.4598             nan     0.1000    0.1692
##      3        1.3552             nan     0.1000    0.1302
##      4        1.2726             nan     0.1000    0.1007
##      5        1.2074             nan     0.1000    0.0847
##      6        1.1518             nan     0.1000    0.0799
##      7        1.1021             nan     0.1000    0.0618
##      8        1.0630             nan     0.1000    0.0678
##      9        1.0208             nan     0.1000    0.0510
##     10        0.9873             nan     0.1000    0.0597
##     20        0.7508             nan     0.1000    0.0256
##     40        0.5215             nan     0.1000    0.0114
##     60        0.3993             nan     0.1000    0.0064
##     80        0.3208             nan     0.1000    0.0037
##    100        0.2625             nan     0.1000    0.0035
##    120        0.2189             nan     0.1000    0.0020
##    140        0.1875             nan     0.1000    0.0013
##    160        0.1613             nan     0.1000    0.0012
##    180        0.1395             nan     0.1000    0.0010
##    200        0.1219             nan     0.1000    0.0003
##    220        0.1069             nan     0.1000    0.0005
##    240        0.0950             nan     0.1000    0.0007
##    260        0.0846             nan     0.1000    0.0005
##    280        0.0754             nan     0.1000    0.0002
##    300        0.0677             nan     0.1000    0.0002
##    320        0.0606             nan     0.1000    0.0003
##    340        0.0544             nan     0.1000    0.0002
##    360        0.0489             nan     0.1000    0.0002
##    380        0.0441             nan     0.1000    0.0000
##    400        0.0398             nan     0.1000    0.0001
##    420        0.0363             nan     0.1000    0.0001
##    440        0.0328             nan     0.1000    0.0001
##    460        0.0298             nan     0.1000    0.0001
##    480        0.0272             nan     0.1000    0.0001
##    500        0.0249             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3094
##      2        1.4171             nan     0.1000    0.2058
##      3        1.2893             nan     0.1000    0.1669
##      4        1.1871             nan     0.1000    0.1257
##      5        1.1095             nan     0.1000    0.1184
##      6        1.0359             nan     0.1000    0.0916
##      7        0.9787             nan     0.1000    0.0843
##      8        0.9253             nan     0.1000    0.0639
##      9        0.8842             nan     0.1000    0.0695
##     10        0.8412             nan     0.1000    0.0600
##     20        0.5691             nan     0.1000    0.0260
##     40        0.3470             nan     0.1000    0.0106
##     60        0.2436             nan     0.1000    0.0052
##     80        0.1839             nan     0.1000    0.0024
##    100        0.1420             nan     0.1000    0.0024
##    120        0.1121             nan     0.1000    0.0015
##    140        0.0900             nan     0.1000    0.0005
##    160        0.0735             nan     0.1000    0.0004
##    180        0.0609             nan     0.1000    0.0002
##    200        0.0510             nan     0.1000    0.0004
##    220        0.0424             nan     0.1000    0.0004
##    240        0.0356             nan     0.1000    0.0005
##    260        0.0298             nan     0.1000    0.0002
##    280        0.0251             nan     0.1000    0.0002
##    300        0.0213             nan     0.1000    0.0000
##    320        0.0183             nan     0.1000    0.0000
##    340        0.0157             nan     0.1000    0.0001
##    360        0.0135             nan     0.1000    0.0001
##    380        0.0117             nan     0.1000    0.0000
##    400        0.0100             nan     0.1000   -0.0000
##    420        0.0087             nan     0.1000    0.0000
##    440        0.0075             nan     0.1000    0.0000
##    460        0.0065             nan     0.1000    0.0000
##    480        0.0057             nan     0.1000    0.0000
##    500        0.0049             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3456
##      2        1.3928             nan     0.1000    0.2457
##      3        1.2397             nan     0.1000    0.1834
##      4        1.1238             nan     0.1000    0.1441
##      5        1.0323             nan     0.1000    0.1171
##      6        0.9582             nan     0.1000    0.0886
##      7        0.9010             nan     0.1000    0.0989
##      8        0.8405             nan     0.1000    0.0789
##      9        0.7903             nan     0.1000    0.0808
##     10        0.7403             nan     0.1000    0.0717
##     20        0.4541             nan     0.1000    0.0237
##     40        0.2477             nan     0.1000    0.0069
##     60        0.1633             nan     0.1000    0.0028
##     80        0.1146             nan     0.1000    0.0019
##    100        0.0837             nan     0.1000    0.0017
##    120        0.0620             nan     0.1000    0.0007
##    140        0.0473             nan     0.1000    0.0008
##    160        0.0364             nan     0.1000    0.0004
##    180        0.0286             nan     0.1000    0.0003
##    200        0.0224             nan     0.1000    0.0003
##    220        0.0177             nan     0.1000    0.0002
##    240        0.0143             nan     0.1000    0.0000
##    260        0.0115             nan     0.1000    0.0001
##    280        0.0093             nan     0.1000    0.0001
##    300        0.0075             nan     0.1000    0.0001
##    320        0.0061             nan     0.1000    0.0001
##    340        0.0050             nan     0.1000    0.0000
##    360        0.0041             nan     0.1000    0.0001
##    380        0.0033             nan     0.1000    0.0000
##    400        0.0027             nan     0.1000    0.0000
##    420        0.0023             nan     0.1000    0.0000
##    440        0.0018             nan     0.1000    0.0000
##    460        0.0015             nan     0.1000    0.0000
##    480        0.0013             nan     0.1000    0.0000
##    500        0.0010             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2300
##      2        1.4606             nan     0.1000    0.1664
##      3        1.3553             nan     0.1000    0.1251
##      4        1.2759             nan     0.1000    0.1052
##      5        1.2113             nan     0.1000    0.0788
##      6        1.1598             nan     0.1000    0.0737
##      7        1.1121             nan     0.1000    0.0733
##      8        1.0667             nan     0.1000    0.0670
##      9        1.0252             nan     0.1000    0.0544
##     10        0.9904             nan     0.1000    0.0609
##     20        0.7527             nan     0.1000    0.0244
##     40        0.5295             nan     0.1000    0.0126
##     60        0.4016             nan     0.1000    0.0059
##     80        0.3252             nan     0.1000    0.0054
##    100        0.2640             nan     0.1000    0.0020
##    120        0.2215             nan     0.1000    0.0018
##    140        0.1879             nan     0.1000    0.0019
##    160        0.1625             nan     0.1000    0.0013
##    180        0.1404             nan     0.1000    0.0010
##    200        0.1226             nan     0.1000    0.0006
##    220        0.1076             nan     0.1000    0.0006
##    240        0.0953             nan     0.1000    0.0004
##    260        0.0851             nan     0.1000    0.0002
##    280        0.0763             nan     0.1000    0.0005
##    300        0.0677             nan     0.1000    0.0002
##    320        0.0606             nan     0.1000    0.0002
##    340        0.0543             nan     0.1000    0.0000
##    360        0.0492             nan     0.1000    0.0001
##    380        0.0445             nan     0.1000    0.0001
##    400        0.0399             nan     0.1000    0.0000
##    420        0.0363             nan     0.1000    0.0001
##    440        0.0328             nan     0.1000    0.0000
##    460        0.0298             nan     0.1000   -0.0000
##    480        0.0271             nan     0.1000    0.0001
##    500        0.0246             nan     0.1000    0.0001
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3047
##      2        1.4163             nan     0.1000    0.2154
##      3        1.2823             nan     0.1000    0.1666
##      4        1.1810             nan     0.1000    0.1171
##      5        1.1056             nan     0.1000    0.1049
##      6        1.0385             nan     0.1000    0.0962
##      7        0.9776             nan     0.1000    0.0926
##      8        0.9200             nan     0.1000    0.0811
##      9        0.8710             nan     0.1000    0.0560
##     10        0.8349             nan     0.1000    0.0535
##     20        0.5684             nan     0.1000    0.0234
##     40        0.3566             nan     0.1000    0.0085
##     60        0.2476             nan     0.1000    0.0055
##     80        0.1838             nan     0.1000    0.0025
##    100        0.1409             nan     0.1000    0.0019
##    120        0.1114             nan     0.1000    0.0018
##    140        0.0885             nan     0.1000    0.0008
##    160        0.0726             nan     0.1000    0.0004
##    180        0.0600             nan     0.1000    0.0004
##    200        0.0496             nan     0.1000    0.0003
##    220        0.0414             nan     0.1000    0.0001
##    240        0.0346             nan     0.1000    0.0002
##    260        0.0291             nan     0.1000    0.0002
##    280        0.0246             nan     0.1000    0.0001
##    300        0.0210             nan     0.1000    0.0001
##    320        0.0178             nan     0.1000    0.0001
##    340        0.0153             nan     0.1000    0.0001
##    360        0.0130             nan     0.1000    0.0001
##    380        0.0111             nan     0.1000    0.0001
##    400        0.0095             nan     0.1000    0.0000
##    420        0.0082             nan     0.1000    0.0001
##    440        0.0071             nan     0.1000    0.0000
##    460        0.0061             nan     0.1000    0.0000
##    480        0.0053             nan     0.1000    0.0000
##    500        0.0046             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3412
##      2        1.3939             nan     0.1000    0.2370
##      3        1.2444             nan     0.1000    0.1740
##      4        1.1366             nan     0.1000    0.1412
##      5        1.0496             nan     0.1000    0.1134
##      6        0.9771             nan     0.1000    0.1113
##      7        0.9064             nan     0.1000    0.1106
##      8        0.8396             nan     0.1000    0.0876
##      9        0.7867             nan     0.1000    0.0722
##     10        0.7410             nan     0.1000    0.0684
##     20        0.4510             nan     0.1000    0.0217
##     40        0.2561             nan     0.1000    0.0080
##     60        0.1680             nan     0.1000    0.0032
##     80        0.1166             nan     0.1000    0.0021
##    100        0.0857             nan     0.1000    0.0008
##    120        0.0637             nan     0.1000    0.0011
##    140        0.0485             nan     0.1000    0.0010
##    160        0.0368             nan     0.1000    0.0005
##    180        0.0284             nan     0.1000    0.0003
##    200        0.0226             nan     0.1000    0.0001
##    220        0.0181             nan     0.1000    0.0001
##    240        0.0144             nan     0.1000    0.0001
##    260        0.0116             nan     0.1000    0.0001
##    280        0.0094             nan     0.1000    0.0001
##    300        0.0076             nan     0.1000    0.0000
##    320        0.0062             nan     0.1000    0.0000
##    340        0.0050             nan     0.1000    0.0000
##    360        0.0041             nan     0.1000    0.0000
##    380        0.0033             nan     0.1000    0.0000
##    400        0.0027             nan     0.1000    0.0000
##    420        0.0022             nan     0.1000    0.0000
##    440        0.0018             nan     0.1000    0.0000
##    460        0.0015             nan     0.1000    0.0000
##    480        0.0012             nan     0.1000    0.0000
##    500        0.0010             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2394
##      2        1.4585             nan     0.1000    0.1619
##      3        1.3564             nan     0.1000    0.1347
##      4        1.2720             nan     0.1000    0.0965
##      5        1.2104             nan     0.1000    0.0853
##      6        1.1557             nan     0.1000    0.0780
##      7        1.1058             nan     0.1000    0.0740
##      8        1.0611             nan     0.1000    0.0691
##      9        1.0168             nan     0.1000    0.0519
##     10        0.9830             nan     0.1000    0.0604
##     20        0.7484             nan     0.1000    0.0310
##     40        0.5185             nan     0.1000    0.0142
##     60        0.3915             nan     0.1000    0.0061
##     80        0.3150             nan     0.1000    0.0034
##    100        0.2584             nan     0.1000    0.0029
##    120        0.2152             nan     0.1000    0.0029
##    140        0.1816             nan     0.1000    0.0020
##    160        0.1561             nan     0.1000    0.0014
##    180        0.1351             nan     0.1000    0.0018
##    200        0.1180             nan     0.1000    0.0009
##    220        0.1027             nan     0.1000    0.0004
##    240        0.0910             nan     0.1000    0.0004
##    260        0.0809             nan     0.1000    0.0003
##    280        0.0718             nan     0.1000    0.0001
##    300        0.0643             nan     0.1000    0.0000
##    320        0.0579             nan     0.1000    0.0001
##    340        0.0520             nan     0.1000    0.0002
##    360        0.0470             nan     0.1000    0.0001
##    380        0.0425             nan     0.1000    0.0002
##    400        0.0382             nan     0.1000    0.0001
##    420        0.0347             nan     0.1000    0.0001
##    440        0.0317             nan     0.1000    0.0001
##    460        0.0290             nan     0.1000   -0.0000
##    480        0.0266             nan     0.1000    0.0001
##    500        0.0243             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3071
##      2        1.4173             nan     0.1000    0.2143
##      3        1.2851             nan     0.1000    0.1568
##      4        1.1880             nan     0.1000    0.1343
##      5        1.1051             nan     0.1000    0.1017
##      6        1.0416             nan     0.1000    0.1038
##      7        0.9770             nan     0.1000    0.0834
##      8        0.9228             nan     0.1000    0.0705
##      9        0.8784             nan     0.1000    0.0764
##     10        0.8321             nan     0.1000    0.0620
##     20        0.5688             nan     0.1000    0.0264
##     40        0.3413             nan     0.1000    0.0081
##     60        0.2422             nan     0.1000    0.0061
##     80        0.1796             nan     0.1000    0.0031
##    100        0.1390             nan     0.1000    0.0017
##    120        0.1095             nan     0.1000    0.0014
##    140        0.0876             nan     0.1000    0.0014
##    160        0.0707             nan     0.1000    0.0005
##    180        0.0582             nan     0.1000    0.0005
##    200        0.0482             nan     0.1000    0.0004
##    220        0.0405             nan     0.1000    0.0004
##    240        0.0341             nan     0.1000    0.0004
##    260        0.0288             nan     0.1000    0.0002
##    280        0.0242             nan     0.1000    0.0002
##    300        0.0205             nan     0.1000    0.0001
##    320        0.0174             nan     0.1000    0.0000
##    340        0.0150             nan     0.1000    0.0000
##    360        0.0130             nan     0.1000    0.0001
##    380        0.0112             nan     0.1000    0.0000
##    400        0.0096             nan     0.1000    0.0000
##    420        0.0083             nan     0.1000    0.0000
##    440        0.0072             nan     0.1000    0.0000
##    460        0.0062             nan     0.1000    0.0000
##    480        0.0054             nan     0.1000    0.0000
##    500        0.0047             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3437
##      2        1.3926             nan     0.1000    0.2427
##      3        1.2400             nan     0.1000    0.1675
##      4        1.1340             nan     0.1000    0.1500
##      5        1.0401             nan     0.1000    0.1336
##      6        0.9565             nan     0.1000    0.1052
##      7        0.8904             nan     0.1000    0.1065
##      8        0.8249             nan     0.1000    0.0730
##      9        0.7787             nan     0.1000    0.0665
##     10        0.7357             nan     0.1000    0.0671
##     20        0.4489             nan     0.1000    0.0233
##     40        0.2482             nan     0.1000    0.0109
##     60        0.1598             nan     0.1000    0.0039
##     80        0.1113             nan     0.1000    0.0019
##    100        0.0823             nan     0.1000    0.0009
##    120        0.0616             nan     0.1000    0.0008
##    140        0.0477             nan     0.1000    0.0006
##    160        0.0369             nan     0.1000    0.0003
##    180        0.0289             nan     0.1000    0.0002
##    200        0.0230             nan     0.1000    0.0002
##    220        0.0183             nan     0.1000    0.0003
##    240        0.0144             nan     0.1000    0.0001
##    260        0.0118             nan     0.1000    0.0001
##    280        0.0095             nan     0.1000    0.0001
##    300        0.0077             nan     0.1000    0.0001
##    320        0.0062             nan     0.1000    0.0001
##    340        0.0051             nan     0.1000    0.0000
##    360        0.0042             nan     0.1000    0.0000
##    380        0.0035             nan     0.1000    0.0000
##    400        0.0028             nan     0.1000    0.0000
##    420        0.0023             nan     0.1000    0.0000
##    440        0.0019             nan     0.1000    0.0000
##    460        0.0016             nan     0.1000    0.0000
##    480        0.0013             nan     0.1000    0.0000
##    500        0.0011             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2347
##      2        1.4579             nan     0.1000    0.1541
##      3        1.3606             nan     0.1000    0.1280
##      4        1.2806             nan     0.1000    0.1112
##      5        1.2102             nan     0.1000    0.0880
##      6        1.1534             nan     0.1000    0.0723
##      7        1.1075             nan     0.1000    0.0687
##      8        1.0645             nan     0.1000    0.0692
##      9        1.0220             nan     0.1000    0.0525
##     10        0.9888             nan     0.1000    0.0568
##     20        0.7583             nan     0.1000    0.0232
##     40        0.5324             nan     0.1000    0.0095
##     60        0.4065             nan     0.1000    0.0092
##     80        0.3272             nan     0.1000    0.0070
##    100        0.2683             nan     0.1000    0.0038
##    120        0.2240             nan     0.1000    0.0018
##    140        0.1894             nan     0.1000    0.0021
##    160        0.1631             nan     0.1000    0.0016
##    180        0.1406             nan     0.1000    0.0010
##    200        0.1223             nan     0.1000    0.0008
##    220        0.1083             nan     0.1000    0.0009
##    240        0.0956             nan     0.1000    0.0003
##    260        0.0847             nan     0.1000    0.0008
##    280        0.0755             nan     0.1000    0.0003
##    300        0.0673             nan     0.1000    0.0002
##    320        0.0609             nan     0.1000   -0.0000
##    340        0.0544             nan     0.1000    0.0002
##    360        0.0485             nan     0.1000    0.0001
##    380        0.0436             nan     0.1000   -0.0001
##    400        0.0396             nan     0.1000    0.0000
##    420        0.0361             nan     0.1000    0.0000
##    440        0.0329             nan     0.1000    0.0001
##    460        0.0298             nan     0.1000   -0.0000
##    480        0.0273             nan     0.1000    0.0001
##    500        0.0248             nan     0.1000   -0.0001
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2981
##      2        1.4195             nan     0.1000    0.2001
##      3        1.2934             nan     0.1000    0.1588
##      4        1.1942             nan     0.1000    0.1372
##      5        1.1098             nan     0.1000    0.1219
##      6        1.0337             nan     0.1000    0.1043
##      7        0.9684             nan     0.1000    0.0804
##      8        0.9183             nan     0.1000    0.0727
##      9        0.8725             nan     0.1000    0.0636
##     10        0.8325             nan     0.1000    0.0587
##     20        0.5627             nan     0.1000    0.0275
##     40        0.3475             nan     0.1000    0.0113
##     60        0.2449             nan     0.1000    0.0049
##     80        0.1823             nan     0.1000    0.0033
##    100        0.1413             nan     0.1000    0.0026
##    120        0.1111             nan     0.1000    0.0014
##    140        0.0888             nan     0.1000    0.0007
##    160        0.0720             nan     0.1000    0.0005
##    180        0.0593             nan     0.1000    0.0006
##    200        0.0491             nan     0.1000    0.0003
##    220        0.0410             nan     0.1000    0.0002
##    240        0.0347             nan     0.1000   -0.0000
##    260        0.0293             nan     0.1000    0.0003
##    280        0.0248             nan     0.1000    0.0001
##    300        0.0211             nan     0.1000    0.0002
##    320        0.0181             nan     0.1000    0.0000
##    340        0.0154             nan     0.1000    0.0001
##    360        0.0132             nan     0.1000    0.0001
##    380        0.0114             nan     0.1000    0.0000
##    400        0.0098             nan     0.1000    0.0000
##    420        0.0085             nan     0.1000    0.0001
##    440        0.0073             nan     0.1000    0.0000
##    460        0.0064             nan     0.1000    0.0000
##    480        0.0056             nan     0.1000    0.0000
##    500        0.0049             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3408
##      2        1.3945             nan     0.1000    0.2427
##      3        1.2433             nan     0.1000    0.1793
##      4        1.1312             nan     0.1000    0.1603
##      5        1.0333             nan     0.1000    0.1170
##      6        0.9589             nan     0.1000    0.1133
##      7        0.8891             nan     0.1000    0.0974
##      8        0.8286             nan     0.1000    0.0784
##      9        0.7791             nan     0.1000    0.0730
##     10        0.7340             nan     0.1000    0.0689
##     20        0.4528             nan     0.1000    0.0260
##     40        0.2505             nan     0.1000    0.0084
##     60        0.1656             nan     0.1000    0.0046
##     80        0.1153             nan     0.1000    0.0026
##    100        0.0841             nan     0.1000    0.0011
##    120        0.0632             nan     0.1000    0.0010
##    140        0.0485             nan     0.1000    0.0007
##    160        0.0376             nan     0.1000    0.0002
##    180        0.0295             nan     0.1000    0.0004
##    200        0.0231             nan     0.1000    0.0003
##    220        0.0184             nan     0.1000    0.0001
##    240        0.0148             nan     0.1000    0.0000
##    260        0.0120             nan     0.1000    0.0001
##    280        0.0096             nan     0.1000    0.0000
##    300        0.0078             nan     0.1000    0.0001
##    320        0.0064             nan     0.1000    0.0000
##    340        0.0053             nan     0.1000    0.0000
##    360        0.0043             nan     0.1000    0.0000
##    380        0.0036             nan     0.1000    0.0000
##    400        0.0029             nan     0.1000   -0.0000
##    420        0.0024             nan     0.1000    0.0000
##    440        0.0020             nan     0.1000    0.0000
##    460        0.0017             nan     0.1000   -0.0000
##    480        0.0014             nan     0.1000    0.0000
##    500        0.0011             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2271
##      2        1.4631             nan     0.1000    0.1681
##      3        1.3570             nan     0.1000    0.1212
##      4        1.2810             nan     0.1000    0.1085
##      5        1.2125             nan     0.1000    0.0783
##      6        1.1604             nan     0.1000    0.0844
##      7        1.1066             nan     0.1000    0.0657
##      8        1.0651             nan     0.1000    0.0683
##      9        1.0221             nan     0.1000    0.0532
##     10        0.9862             nan     0.1000    0.0485
##     20        0.7524             nan     0.1000    0.0225
##     40        0.5298             nan     0.1000    0.0148
##     60        0.4042             nan     0.1000    0.0076
##     80        0.3205             nan     0.1000    0.0033
##    100        0.2633             nan     0.1000    0.0026
##    120        0.2193             nan     0.1000    0.0019
##    140        0.1842             nan     0.1000    0.0019
##    160        0.1573             nan     0.1000    0.0025
##    180        0.1348             nan     0.1000    0.0008
##    200        0.1179             nan     0.1000    0.0003
##    220        0.1044             nan     0.1000    0.0007
##    240        0.0928             nan     0.1000    0.0004
##    260        0.0822             nan     0.1000    0.0004
##    280        0.0730             nan     0.1000    0.0003
##    300        0.0658             nan     0.1000    0.0002
##    320        0.0591             nan     0.1000    0.0002
##    340        0.0531             nan     0.1000    0.0002
##    360        0.0476             nan     0.1000    0.0001
##    380        0.0431             nan     0.1000    0.0001
##    400        0.0390             nan     0.1000    0.0001
##    420        0.0353             nan     0.1000    0.0000
##    440        0.0319             nan     0.1000    0.0000
##    460        0.0291             nan     0.1000    0.0000
##    480        0.0267             nan     0.1000    0.0000
##    500        0.0244             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3043
##      2        1.4180             nan     0.1000    0.2071
##      3        1.2870             nan     0.1000    0.1623
##      4        1.1862             nan     0.1000    0.1347
##      5        1.1019             nan     0.1000    0.1003
##      6        1.0386             nan     0.1000    0.1013
##      7        0.9758             nan     0.1000    0.0893
##      8        0.9186             nan     0.1000    0.0846
##      9        0.8657             nan     0.1000    0.0665
##     10        0.8233             nan     0.1000    0.0618
##     20        0.5547             nan     0.1000    0.0231
##     40        0.3398             nan     0.1000    0.0115
##     60        0.2366             nan     0.1000    0.0060
##     80        0.1774             nan     0.1000    0.0021
##    100        0.1369             nan     0.1000    0.0017
##    120        0.1092             nan     0.1000    0.0017
##    140        0.0879             nan     0.1000    0.0009
##    160        0.0718             nan     0.1000    0.0005
##    180        0.0587             nan     0.1000    0.0003
##    200        0.0485             nan     0.1000    0.0003
##    220        0.0403             nan     0.1000    0.0003
##    240        0.0339             nan     0.1000    0.0002
##    260        0.0285             nan     0.1000    0.0001
##    280        0.0242             nan     0.1000    0.0002
##    300        0.0207             nan     0.1000    0.0000
##    320        0.0177             nan     0.1000    0.0001
##    340        0.0153             nan     0.1000    0.0001
##    360        0.0131             nan     0.1000    0.0000
##    380        0.0114             nan     0.1000    0.0000
##    400        0.0099             nan     0.1000    0.0000
##    420        0.0086             nan     0.1000    0.0001
##    440        0.0074             nan     0.1000    0.0000
##    460        0.0064             nan     0.1000    0.0000
##    480        0.0055             nan     0.1000   -0.0000
##    500        0.0048             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3481
##      2        1.3892             nan     0.1000    0.2328
##      3        1.2436             nan     0.1000    0.1770
##      4        1.1318             nan     0.1000    0.1610
##      5        1.0329             nan     0.1000    0.1158
##      6        0.9597             nan     0.1000    0.1155
##      7        0.8890             nan     0.1000    0.0926
##      8        0.8311             nan     0.1000    0.0785
##      9        0.7815             nan     0.1000    0.0622
##     10        0.7414             nan     0.1000    0.0608
##     20        0.4574             nan     0.1000    0.0292
##     40        0.2518             nan     0.1000    0.0091
##     60        0.1649             nan     0.1000    0.0043
##     80        0.1160             nan     0.1000    0.0025
##    100        0.0841             nan     0.1000    0.0017
##    120        0.0629             nan     0.1000    0.0012
##    140        0.0482             nan     0.1000    0.0005
##    160        0.0367             nan     0.1000    0.0001
##    180        0.0286             nan     0.1000    0.0003
##    200        0.0225             nan     0.1000    0.0001
##    220        0.0178             nan     0.1000    0.0001
##    240        0.0142             nan     0.1000   -0.0000
##    260        0.0116             nan     0.1000    0.0002
##    280        0.0094             nan     0.1000    0.0000
##    300        0.0077             nan     0.1000    0.0001
##    320        0.0062             nan     0.1000    0.0000
##    340        0.0051             nan     0.1000    0.0000
##    360        0.0041             nan     0.1000    0.0000
##    380        0.0034             nan     0.1000    0.0000
##    400        0.0028             nan     0.1000    0.0000
##    420        0.0023             nan     0.1000    0.0000
##    440        0.0019             nan     0.1000    0.0000
##    460        0.0015             nan     0.1000    0.0000
##    480        0.0013             nan     0.1000    0.0000
##    500        0.0010             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.3379
##      2        1.3965             nan     0.1000    0.2428
##      3        1.2432             nan     0.1000    0.1915
##      4        1.1253             nan     0.1000    0.1428
##      5        1.0356             nan     0.1000    0.1401
##      6        0.9502             nan     0.1000    0.1094
##      7        0.8832             nan     0.1000    0.0957
##      8        0.8240             nan     0.1000    0.0785
##      9        0.7752             nan     0.1000    0.0650
##     10        0.7347             nan     0.1000    0.0677
##     20        0.4534             nan     0.1000    0.0248
##     40        0.2521             nan     0.1000    0.0082
##     60        0.1696             nan     0.1000    0.0053
##     80        0.1187             nan     0.1000    0.0022
##    100        0.0874             nan     0.1000    0.0016
##    120        0.0660             nan     0.1000    0.0011
##    140        0.0507             nan     0.1000    0.0007
##    160        0.0394             nan     0.1000    0.0007
##    180        0.0315             nan     0.1000    0.0004
##    200        0.0249             nan     0.1000    0.0002
##    220        0.0200             nan     0.1000    0.0002
##    240        0.0161             nan     0.1000    0.0002
##    260        0.0130             nan     0.1000    0.0001
##    280        0.0107             nan     0.1000    0.0001
##    300        0.0087             nan     0.1000    0.0001
##    320        0.0071             nan     0.1000    0.0000
##    340        0.0059             nan     0.1000    0.0000
##    360        0.0048             nan     0.1000    0.0000
##    380        0.0040             nan     0.1000    0.0000
##    400        0.0033             nan     0.1000    0.0000
##    420        0.0028             nan     0.1000    0.0000
##    440        0.0023             nan     0.1000    0.0000
##    460        0.0019             nan     0.1000    0.0000
##    480        0.0016             nan     0.1000    0.0000
##    500        0.0013             nan     0.1000    0.0000
```

```r
pred_gbm <- predict(modelFit_gbm, newdata=validating_sub)

confusionMatrix(validating$classe,pred_gbm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    0    0    0    1
##          B    1  758    0    0    0
##          C    0    3  679    2    0
##          D    0    0    8  635    0
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9959          
##                  95% CI : (0.9934, 0.9977)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9948          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9961   0.9884   0.9953   0.9986
## Specificity            0.9996   0.9997   0.9985   0.9976   0.9997
## Pos Pred Value         0.9991   0.9987   0.9927   0.9876   0.9986
## Neg Pred Value         0.9996   0.9991   0.9975   0.9991   0.9997
## Prevalence             0.2845   0.1940   0.1751   0.1626   0.1838
## Detection Rate         0.2842   0.1932   0.1731   0.1619   0.1835
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9994   0.9979   0.9934   0.9964   0.9992
```

## Train a model - Random Forest



```r
set.seed(39785)

modelFit_rf <- train(classe~., method='rf', data=training_sub, tunelength=30,
                      trControl=trainControl(method='cv', number=5, search = 'random'))

pred_rf <- predict(modelFit_rf, newdata=validating_sub)

confusionMatrix(validating$classe,pred_rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    0    0    0    1
##          B    1  757    1    0    0
##          C    0    4  680    0    0
##          D    0    0   10  633    0
##          E    0    0    0    2  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9952          
##                  95% CI : (0.9924, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9939          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9947   0.9841   0.9969   0.9986
## Specificity            0.9996   0.9994   0.9988   0.9970   0.9994
## Pos Pred Value         0.9991   0.9974   0.9942   0.9844   0.9972
## Neg Pred Value         0.9996   0.9987   0.9966   0.9994   0.9997
## Prevalence             0.2845   0.1940   0.1761   0.1619   0.1835
## Detection Rate         0.2842   0.1930   0.1733   0.1614   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9994   0.9971   0.9914   0.9969   0.9990
```


## Compare predictions from two models


```r
confusionMatrix(pred_gbm,pred_rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  758    3    0    0
##          C    0    3  684    0    0
##          D    0    0    4  634    0
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9972         
##                  95% CI : (0.995, 0.9986)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9965         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9961   0.9899   0.9984   1.0000
## Specificity            1.0000   0.9991   0.9991   0.9988   0.9997
## Pos Pred Value         1.0000   0.9961   0.9956   0.9937   0.9986
## Neg Pred Value         1.0000   0.9991   0.9978   0.9997   1.0000
## Prevalence             0.2845   0.1940   0.1761   0.1619   0.1835
## Detection Rate         0.2845   0.1932   0.1744   0.1616   0.1835
## Detection Prevalence   0.2845   0.1940   0.1751   0.1626   0.1838
## Balanced Accuracy      1.0000   0.9976   0.9945   0.9986   0.9998
```


## Present results for test sample


```r
#Predict for test set with gbm
predict(modelFit_gbm, newdata=pml_testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
#Predict for test set with rf
predict(modelFit_rf, newdata=pml_testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
