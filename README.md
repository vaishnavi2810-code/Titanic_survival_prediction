# Titanic_survival_prediction
#### A machine learning model for predicting the survival of passengers aboard RMS Titanic, a British passenger liner that sank in the North Atlantic Ocean.
### The first step is to add libraries
```r
library(dplyr)
library(tidyr)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
library(ModelMetrics)
```
### Import the dataset
```r
Titanic_Data <- read.csv("/home/vaishnavi/Project/dataset.csv")
glimpse(Titanic_Data)
View(Titanic_Data)
```

