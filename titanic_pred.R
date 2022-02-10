library(dplyr)
library(tidyr)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
library(ModelMetrics)



Titanic_Data <- read.csv("/home/vaishnavi/Project/dataset.csv")
glimpse(Titanic_Data)
View(Titanic_Data)
any(Titanic_Data=="?")##True that means there exist some values as such "?"
which(Titanic_Data=="?",arr.ind=TRUE)
str(Titanic_Data)
colSums(is.na(Titanic_Data))
head(Titanic_Data)

#removing all target class NA valued rows from whole dataset
Titanic_Data <- Titanic_Data %>% drop_na('Survived')
#removing duplicate columns
Titanic_Data <- select(Titanic_Data,-c('Pclass', 'Age', 'Name'))
colSums(is.na(Titanic_Data))



#removing na values
library(ggplot2)
ggplot(Titanic_Data,aes(x=Age_wiki))+geom_bar()
mean(as.numeric(Titanic_Data$Age_wiki),na.rm = TRUE) #29.00 approx

Titanic_Data$Age_wiki <- ifelse(is.na(Titanic_Data$Age_wiki),
                                      as.character(29.00),
                                Titanic_Data$Age_wiki)
summary(Titanic_Data$Class)
str(Titanic_Data$Class)

Titanic_Data$Class <- ifelse(is.na(Titanic_Data$Class),
                             median(Titanic_Data$Class,na.rm = TRUE),
                             Titanic_Data$Class)


#removing useless columns
str(Titanic_Data$WikiId)
str(Titanic_Data$Body)
Titanic_Data <- select(Titanic_Data,-c('WikiId'))
Titanic_Data <- select(Titanic_Data,-c('Body'))

table(Titanic_Data$Survived) #1 for survived , 0 for dead 

head(Titanic_Data$Ticket)
Titanic_Data$Age_wiki <- as.numeric(Titanic_Data$Age_wiki)

Titanic_Data$Age_wiki <- cut(Titanic_Data$Age_wiki , 
                             breaks=c(0,10,20,30,40,50,60,70,100),
                             labels = c("0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-100"))
summary(Titanic_Data$Age_wiki)                            

colSums(is.na(Titanic_Data))
#copying dataset
Titanic_c = Titanic_Data[,c('Class','Sex','SibSp','Parch','Fare',
                            'Age_wiki','Embarked','Survived')]
#View(Titanic_c) 

train_row_id <- sample(1:nrow(Titanic_c),size = ceiling(0.7*nrow(Titanic_c)))
#View(train_row_id)

X_train <- Titanic_c[train_row_id, ]
x_test <- Titanic_c[-train_row_id, ]
#removing col to be predicted from test dataset
x_test1 <- x_test[,-8] 


#data exploration and visualisation
library(ggplot2)
ggplot(Titanic_c,aes(x=Sex))+geom_bar()
table(Titanic_c$Sex)
ggplot(Titanic_c,aes(x=Survived,fill=Sex))+geom_histogram(bins = 5)

summary(Titanic_c$Age_wiki)
ggplot(Titanic_c,aes(x=Survived,fill=Age_wiki))+geom_histogram(bins = 6)


board_city <- unique(Titanic_Data$Boarded)
board_city[1]
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


#model building
model <- rpart(Survived~.,data = X_train,method="class")
summary(model)
View(model)

#model testing
model_pr <- predict(model,x_test1)

plot(model)
plot(model,uniform = TRUE,main="Titanic Survival")
text(model,use.n = TRUE, all = TRUE, cex=.8)

View(model_pr)
model_pr <- ifelse(model_pr[,1] <= model_pr[,2],1,0)
View(model_pr)
str(model_pr)
prediction <- table(model_pr, x_test$Survived)

acc2<-sum(diag(prediction))/sum(prediction)*100
print(acc2)


#creating class variables
titanic_class<-ifelse(Titanic_c$Survived==0,"No","Yes");
titanic_2<-data.frame(Titanic_c,titanic_class)
#using naive bayesian
set.seed(2)
id<-sample(2,nrow(titanic_2),prob=c(.7,.3),replace=TRUE)
print(id)
titanic_train<-titanic_2[id==1,]
titanic_test<-titanic_2[id==2,]
print(titanic_train)
print(titanic_test)
library(e1071)


model<-naiveBayes(titanic_class~.,titanic_train)
print(model)
#predict model
pmodel<-predict(model,titanic_test)
#plot confusion matrix
prediction <- table(pmodel,titanic_test$titanic_class)
acc2<-sum(diag(prediction))/sum(prediction)*100
print(acc2)
str(pmodel)
