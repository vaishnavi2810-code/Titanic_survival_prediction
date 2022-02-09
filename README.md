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
<img src="https://user-images.githubusercontent.com/65387125/153166584-4c470fe1-bd67-4b2e-9292-cb5859d09ca2.png"></img>
>This dataset contains total 21 columns as above.
### Next step is to look for NA values or any unnecessary symbols 
```r
any(Titanic_Data=="?")##True that means there exist some values as such "?"
which(Titanic_Data=="?",arr.ind=TRUE)
```
<img src="https://user-images.githubusercontent.com/65387125/153176180-e421a482-857c-403d-9572-65bb38c5e304.png"></img>
>The above code is for the verification if there exists any "?" in the dataset 
```r
colSums(is.na(Titanic_Data))
head(Titanic_Data)
```
<img src="https://user-images.githubusercontent.com/65387125/153176659-640f4e37-c5dd-4186-92f9-c28f7d6c0d6f.png"></img>
>The above code is to check of NA values. It displays the number of NA values in every column
