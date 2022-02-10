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
### Next step is to remove the unecessary columns.
```r
str(Titanic_Data$WikiId)
str(Titanic_Data$Body)
```
<img src="https://user-images.githubusercontent.com/65387125/153343170-7588d9c0-1980-42ed-9d95-576c68ffbd5f.png"></img>
>By this information we can know that WikiId and Body columns has no importance in the dataset, so we can remove those columns.
```r
Titanic_Data <- select(Titanic_Data,-c('WikiId'))
Titanic_Data <- select(Titanic_Data,-c('Body'))
```
```r
table(Titanic_Data$Survived) #1 for survived , 0 for dead 
```
<img src="https://user-images.githubusercontent.com/65387125/153343556-2cf49f27-888e-4635-aa6b-8faaf8a03a37.png"></img>
>This lists the number of passenegers in the target class, 1 is for survived and 0 is for not-survived.
```r
head(Titanic_Data$Ticket)
Titanic_Data$Age_wiki <- as.numeric(Titanic_Data$Age_wiki)
```
>Convert Age_wiki to numeric data type.
```r
Titanic_Data$Age_wiki <- cut(Titanic_Data$Age_wiki , breaks=c(0,10,20,30,40,50,60,70,100), labels = c("0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-100"))
summary(Titanic_Data$Age_wiki)                            
colSums(is.na(Titanic_Data))
```
<img src="https://user-images.githubusercontent.com/65387125/153348264-0d68706a-e454-428e-89fc-6207c559b256.png"></img>
>The above code breaks the age into range of values as you can see in the image. Finally all the NA values are removed.
### Training and Testing dataset
```r
Titanic_c = Titanic_Data[,c('Class','Sex','SibSp','Parch','Fare','Age_wiki','Embarked','Survived')]
```
>Copy this dataset as Titanic_c to divide it into traning and testing part.
```r
train_row_id <- sample(1:nrow(Titanic_c),size = ceiling(0.7*nrow(Titanic_c)))
#View(train_row_id)
X_train <- Titanic_c[train_row_id, ]
x_test <- Titanic_c[-train_row_id, ]
#removing col to be predicted from test dataset
x_test1 <- x_test[,-8] 
```
>divide the dataset into training and testing part and remove the target class from it.
### Data exploration and visualization
#### This step is important because it gives us a better understanding of the dataset. The better we know the data, the better our analysis will be.
```r
library(ggplot2)
ggplot(Titanic_c,aes(x=Sex))+geom_bar()
```
<img src="https://user-images.githubusercontent.com/65387125/153350329-b36fba93-54ae-4702-931c-9089b1713ffb.png"></img>
```r
table(Titanic_c$Sex)
```
<img src="https://user-images.githubusercontent.com/65387125/153350457-44848298-f787-4f55-aecb-e8cb3c1ca01f.png"></img>
```r
ggplot(Titanic_c,aes(x=Survived,fill=Sex))+geom_histogram(bins = 5)
```
<img src="https://user-images.githubusercontent.com/65387125/153350597-306893af-dced-49e3-b2b2-bc59eeb1dc61.png"></img>
```r
summary(Titanic_c$Age_wiki)
```
<img src="https://user-images.githubusercontent.com/65387125/153350836-946d0b68-b5a8-491e-9f82-cd9e425726d0.png"></img>
```r
ggplot(Titanic_c,aes(x=Survived,fill=Age_wiki))+geom_histogram(bins = 6)
```
<img src="https://user-images.githubusercontent.com/65387125/153351051-95abe4fd-6337-4d64-921d-fcbb773fd79b.png"></img>
```r
board_city <- unique(Titanic_Data$Boarded)
board_city[1]
```
<img src="https://user-images.githubusercontent.com/65387125/153351299-e03cfede-4d83-4358-a006-f2616ad3fd72.png"></img>
```r
count1=0
count2=0
count3=0
count4=0
count5=0
Titanic_Data$Boarded[1]
for(i in 1:nrow(Titanic_Data)){
  if(Titanic_Data$Boarded[i]==board_city[1]){
    count1=count1+1
  }
  if(Titanic_Data$Boarded[i]==board_city[2]){
    count2=count2+1
  }
  if(Titanic_Data$Boarded[i]==board_city[3]){
    count3=count3+1
  }
  if(Titanic_Data$Boarded[i]==board_city[4]){
    count4=count4+1
  }
  if(Titanic_Data$Boarded[i]==board_city[5]){
    count5=count5+1
  }
}
count_board <- c(count1,count2,count3,count4,count5)
pie(count_board,labels = board_city,
    col = rainbow(5))
```
<img src="https://user-images.githubusercontent.com/65387125/153351589-ec93cfd6-0ade-43aa-86be-c3928b169213.png"></img>
### Model building - Decision Tree 
#### First model that we build is a decision tree model.
```r
#model building
model <- rpart(Survived~.,data = X_train,method="class")
View(model)
```
<img src="https://user-images.githubusercontent.com/65387125/153364436-de0c6469-2316-4430-b21b-ec2618e9f2b9.png"></img>
#### Testing the model
```r
#model testing
model_pr <- predict(model,x_test1)
plot(model)
plot(model,uniform = TRUE,main="Titanic Survival")
text(model,use.n = TRUE, all = TRUE, cex=.8)
View(model_pr)
```
<img src="https://user-images.githubusercontent.com/65387125/153394222-cad27df2-4807-4bc4-a4c7-b56211182196.png"></img>
```r
model_pr <- ifelse(model_pr[,1] <= model_pr[,2],1,0)
View(model_pr)
str(model_pr)
prediction <- table(model_pr, x_test$Survived)
```
<img src="https://user-images.githubusercontent.com/65387125/153394665-7af9043e-ae51-48cb-9611-704168eb825c.png"></img><br>
<img src="https://user-images.githubusercontent.com/65387125/153394813-eb5d199a-db9d-4e04-942b-2ca0e2cb6aba.png"></img>
```r
acc2<-sum(diag(prediction))/sum(prediction)*100
print(acc2)
```
<img src="https://user-images.githubusercontent.com/65387125/153395530-2c706897-676f-499b-b8ab-7b7b74dbf008.png"></img>
>The above code gives us the accuracy of the model which is 80.89%.
### Model building - Navie Bayesian
### Now we will build another model which is Navie Bayesian and we will compare it's performance with the decision tree model.
### First step is to define the class variables.
```r
#creating class variables
titanic_class<-ifelse(Titanic_c$Survived==0,"No","Yes");
titanic_2<-data.frame(Titanic_c,titanic_class)
```
### Then we will set the seed and divide our dataset into training and testing set.
```r
#using naive bayesian
set.seed(2)
id<-sample(2,nrow(titanic_2),prob=c(.7,.3),replace=TRUE)
print(id)
```
<img src="https://user-images.githubusercontent.com/65387125/153396451-2945183e-38da-4b30-8908-a1d70e20e105.png"></img>
```r
titanic_train<-titanic_2[id==1,]
titanic_test<-titanic_2[id==2,]
print(titanic_train)
print(titanic_test)
```
### Building the Naive Bayesian model.
```r
library(e1071)
model<-naiveBayes(titanic_class~.,titanic_train)
print(model)
```
<img src="https://user-images.githubusercontent.com/65387125/153396893-f081b352-29ca-4e36-9592-d058d7c9c260.png"></img><br>
<img src="https://user-images.githubusercontent.com/65387125/153397018-8c030df6-7f4e-4592-935c-8741f55f8a9b.png"></img>
### Predicting the model
```r
#predict model
pmodel<-predict(model,titanic_test)
```
### Plot confusion matrix
```r
#plot confusion matrix
prediction <- table(pmodel,titanic_test$titanic_class)
acc2<-sum(diag(prediction))/sum(prediction)*100
print(acc2)
str(pmodel)
```
<img src="https://user-images.githubusercontent.com/65387125/153397470-dda63b4d-fd11-4d46-af46-d2e2b6ab082a.png"></img>
>By using Navie Bayesian, we are getting the accuracy 98.89%. So we can say that Naive Bayesian is a better approach for Titanic Survival Prediction.
