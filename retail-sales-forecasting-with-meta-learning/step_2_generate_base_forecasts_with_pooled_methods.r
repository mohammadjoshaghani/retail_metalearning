
### step 2, generate base forecasts with pooled methods
library(reshape2)
library(data.table)
library(doParallel)
load('IRI_storeitem_datasets')


get_predictors<-function(i,data,start,end,h,lags){
Y<-unlist(data$unit[i,start:ncol(data$unit),with=FALSE])
Y.mean=mean(log(Y[1:(length(Y)-h)]+1))
price<-unlist(data$price[i,start:ncol(data$unit),with=FALSE])
price[price==Inf]<-mean(price[price!=Inf])
price[price<=0]<-mean(price[price>0])
price.mean=log(mean(price[1:(length(price)-h)]))

F<-unlist(data$F[i,start:ncol(data$unit),with=FALSE])
D<-unlist(data$D[i,start:ncol(data$unit),with=FALSE])
holiday<-unlist(data$holiday[i,start:ncol(data$unit),with=FALSE])
Y.lags<-c()
price.lags<-c()
F.lags<-c()
D.lags<-c()

if (lags>0){
Y.lags<- matrix(Y[1],length(Y),lags+h)
for(j in 1:(lags+h)){
Y.lags[(1+j):length(Y),j]<-unlist(data$unit[i,start:(ncol(data$unit)-j),with=FALSE])
}
Y.lags<-log(Y.lags+1)-Y.mean

price.lags<- matrix(price[1],length(Y),lags)
for(j in 1:(lags)){
price.lags[(1+j):length(Y),j]<-unlist(data$price[i,start:(ncol(data$unit)-j),with=FALSE])
}
price.lags[price.lags==Inf]<-mean(price.lags[price.lags!=Inf])
price.lags[price.lags<=0]<-mean(price.lags[price.lags>0])
price.lags<-log(price.lags)-price.mean

F.lags<- matrix(F[1],length(Y),(lags))
for(j in 1:(lags)){
F.lags[(1+j):length(Y),j]<-unlist(data$F[i,start:(ncol(data$unit)-j),with=FALSE])
}

D.lags<- matrix(D[1],length(Y),(lags))
for(j in 1:(lags)){
D.lags[(1+j):length(Y),j]<-unlist(data$D[i,start:(ncol(data$unit)-j),with=FALSE])
}

H.lags<- matrix(holiday[1],length(Y),(lags))
for(j in 1:(lags)){
H.lags[(1+j):length(Y),j]<-unlist(data$holiday[i,start:(ncol(data$unit)-j),with=FALSE])
}

}

price<-log(price)-price.mean
list(X=cbind(price,price.lags,F,F.lags,D,D.lags,holiday,H.lags),Y=cbind(log(Y+1)-Y.mean,Y.lags),Y_M=Y.mean)
}

######## xgboost 

xgboost_Dir_forecast<-function(data,start,end,h,lags,get_predictors){
library(xgboost)
library(data.table)
results<-matrix(0,nrow(data$unit),h)

X<-lapply(1:nrow(data$unit),get_predictors,data,start,end,h,lags)
X.tr<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},lags+1,48))
X.ts<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},49,55))

for (H in 1:h){
predictors<-c((H+1):(H+lags),(lags+h+2):ncol(X.tr))
x.dtrain <- xgb.DMatrix(X.tr[,predictors],label = X.tr[,1] )
x.dtest  <- xgb.DMatrix(X.ts[,predictors],label = X.ts[,1] )
bst <- xgb.train(data = x.dtrain,nrounds= 1000,eta = 0.02)
forecasts=exp(predict(bst,x.dtest)+X.ts[,ncol(X.ts)])-1
h1<-seq(H,nrow(X.ts),by=h)
results[,H]<-forecasts[h1]
}
results
}

cl <- makeCluster(getOption("cl.cores", 5))
Xgboost_forecast_3<-parLapply(cl,IRI_storeitem_datasets,xgboost_Dir_forecast,7,54,7,3,get_predictors)
Xgboost_forecast_7<-parLapply(cl,IRI_storeitem_datasets,xgboost_Dir_forecast,7,54,7,7,get_predictors)
stopCluster(cl)
save(Xgboost_forecast_3,file='./base forecasters/Xgboost_forecast_3')
save(Xgboost_forecast_7,file='./base forecasters/Xgboost_forecast_7')

#########################################################   lasso

Lasso_Dir_forecast<-function(data,start,end,h,lags,get_predictors){
library(glmnet)
library(data.table)
results<-matrix(0,nrow(data$unit),h)

X<-lapply(1:nrow(data$unit),get_predictors,data,start,end,h,lags)
X.tr<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},lags+1,48))
X.ts<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},49,55))

for (H in 1:h){
predictors<-c((H+1):(H+lags),(lags+h+2):ncol(X.tr))
bst <- cv.glmnet(X.tr[,predictors],X.tr[,1])
forecasts=exp(predict(bst,X.ts[,predictors],lambda=bst$lambda.min)+X.ts[,ncol(X.ts)])-1
h1<-seq(H,nrow(X.ts),by=h)
results[,H]<-forecasts[h1]
}
results
}

cl <- makeCluster(getOption("cl.cores", 5))
Lasso_Dir_3<-parLapply(cl,IRI_storeitem_datasets,Lasso_Dir_forecast,7,54,7,3,get_predictors)
Lasso_Dir_7<-parLapply(cl,IRI_storeitem_datasets,Lasso_Dir_forecast,7,54,7,7,get_predictors)
stopCluster(cl)
save(Lasso_Dir_3,file='./base forecasters/Lasso_Dir_3')
save(Lasso_Dir_7,file='./base forecasters/Lasso_Dir_7')


####################### Random forest

Ranger_Dir_forecast<-function(data,start,end,h,lags,get_predictors){
library(ranger)
library(data.table)
results<-matrix(0,nrow(data$unit),h)

X<-lapply(1:nrow(data$unit),get_predictors,data,start,end,h,lags)
X.tr<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},lags+1,48))
X.ts<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},49,55))

for (H in 1:h){
predictors<-c(1,(H+1):(H+lags),(lags+h+2):ncol(X.tr))
ranger.data<-as.data.frame(X.tr[,predictors])
bst <- ranger(V1~.,data=ranger.data)
forecasts=exp(predict(bst,as.data.frame(X.ts[,predictors]))$predictions+X.ts[,ncol(X.ts)])-1
h1<-seq(H,nrow(X.ts),by=h)
results[,H]<-forecasts[h1]
}
results
}

cl <- makeCluster(getOption("cl.cores", 5))
Ranger_Dir_3<-parLapply(cl,IRI_storeitem_datasets,Ranger_Dir_forecast,7,54,7,3,get_predictors)
Ranger_Dir_7<-parLapply(cl,IRI_storeitem_datasets,Ranger_Dir_forecast,7,54,7,7,get_predictors)
stopCluster(cl)
save(Ranger_Dir_3,file='./base forecasters/Ranger_Dir_3')
save(Ranger_Dir_7,file='./base forecasters/Ranger_Dir_7')

############# ELM 

ELM_pool_Dir_forecast<-function(data,start,end,h,lags,get_predictors){
library(elmNNRcpp)
library(data.table)
results<-matrix(0,nrow(data$unit),h)

X<-lapply(1:nrow(data$unit),get_predictors,data,start,end,h,lags)
X.tr<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},lags+1,48))
X.ts<-do.call(rbind,lapply(X,function(x,s,w){cbind(x$Y[s:w,],x$X[s:w,],x$Y_M)},49,55))

for (H in 1:h){
predictors<-c((H+1):(H+lags),(lags+h+2):ncol(X.tr))
elmX<-elm_train(X.tr[,predictors],as.matrix(X.tr[,1]),nhid =round(nrow(results)/100),actfun='purelin')
forecasts=exp(elm_predict(elmX,X.ts[,predictors])+X.ts[,ncol(X.ts)])-1
h1<-seq(H,nrow(X.ts),by=h)
results[,H]<-forecasts[h1]
}
results
}

cl <- makeCluster(getOption("cl.cores", 5))
ELM_pool_Dir_3<-parLapply(cl,IRI_storeitem_datasets,ELM_pool_Dir_forecast,7,54,7,3,get_predictors)
ELM_pool_Dir_7<-parLapply(cl,IRI_storeitem_datasets,ELM_pool_Dir_forecast,7,54,7,7,get_predictors)
stopCluster(cl)
save(ELM_pool_Dir_3,file='./base forecasters/ELM_pool_Dir_3')
save(ELM_pool_Dir_7,file='./base forecasters/ELM_pool_Dir_7')

























