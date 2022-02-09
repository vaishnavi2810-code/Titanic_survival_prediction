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
>The above code is for the verification if there exists any "?" in the dataset.
```r
colSums(is.na(Titanic_Data))
head(Titanic_Data)
```
<img src="https://user-images.githubusercontent.com/65387125/153176659-640f4e37-c5dd-4186-92f9-c28f7d6c0d6f.png"></img>
>The above code is to check of NA values. It displays the number of NA values in every column.
### Next step is to remove NA values
```r
#removing all target class NA valued rows from whole dataset
Titanic_Data <- Titanic_Data %>% drop_na('Survived')
#removing duplicate columns
Titanic_Data <- select(Titanic_Data,-c('Pclass', 'Age', 'Name'))
colSums(is.na(Titanic_Data))
```
<img src="https://user-images.githubusercontent.com/65387125/153180917-63f8c8c7-31c9-4712-ab50-f733463187fa.png"></img>
>The above code removes all NA values from the target class which is Survived. Duplicate columns are aslo removed, the columns Pclass, Age, Name are same as the columns Class, Age_wiki and Name_wiki respectively.
```r
library(ggplot2)
ggplot(Titanic_Data,aes(x=Age_wiki))+geom_bar()
mean(as.numeric(Titanic_Data$Age_wiki),na.rm = TRUE) #29.00 approx
```
<img src="https://user-images.githubusercontent.com/65387125/153183844-cf976480-388c-43f9-ae80-fe7679cd73d8.png"></img>
<img src="https://user-images.githubusercontent.com/65387125/153182819-04986faa-42e3-4325-ad6d-548756e2fc24.png"></img>
>Plot a bar graph to check which is best aggregate to replace NA values in Age_wiki column. Now find the mean of Age_wiki column to replace NA values by mean.
```r
Titanic_Data$Age_wiki <- ifelse(is.na(Titanic_Data$Age_wiki), as.character(29.00), Titanic_Data$Age_wiki)
```
>Replacing NA values with mean.
```r
summary(Titanic_Data$Class)
str(Titanic_Data$Class)
```
<img src="https://user-images.githubusercontent.com/65387125/153184779-3dce156b-d05d-494f-ba6b-a3ccb053199d.png"></img>
>View the summary of Class column to find out the best aggregate for replacing NA values.
```r
Titanic_Data$Class <- ifelse(is.na(Titanic_Data$Class), median(Titanic_Data$Class,na.rm = TRUE), Titanic_Data$Class)
```
> Replace NA values with median.

