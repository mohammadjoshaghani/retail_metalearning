
### step 2, generate individual forecasts
library(reshape2)
library(data.table)
library(doParallel)
load('Z:/project/meta forecast/IRI_storeitem_datasets')

##ETS forecasts

ETS_forecast<-function(data,start,end,h){
library(forecast)
library(data.table)
results<-matrix(0,nrow(data$unit),h)
for( i in 1:nrow(data$unit)){
data.ts<-ts(unlist(data$unit[i,start:end,with=FALSE]))
model.ets<- ets(data.ts+1,lambda=0)
pred<-forecast(model.ets,h=h,lambda=0)$mean-1
data.max<-max(data.ts)
pred[pred>(2*data.max)]<- data.max ## sometimes ets give abnormal predictions when using log transforms
pred[pred<0]<- 0
results[i,]<- pred
}
results
}

cl <- makeCluster(getOption("cl.cores", length(IRI_storeitem_datasets)))
ETS_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,ETS_forecast,7,54,7)
stopCluster(cl)
save(ETS_forecast_resutls,file='./base forecasters/ETS_forecast_resutls')


get_predictors<-function(i,data,start,end,h,lags){   ## data preparation for regressions
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
}
price<-log(price)-price.mean
list(X=cbind(price,price.lags,F,F.lags,D,D.lags,holiday),Y=Y.lags)
}

####################  ADL forecasts

sku_pre_Dir<-function(i,data,start,end,h,lags,get_predictors){
Y<-unlist(data$unit[i,start:ncol(data$unit),with=FALSE])
Y.max<- max(Y)
Y.mean=mean(log(Y[1:(length(Y)-h)]+1))
Y.tr<-log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean
X<-get_predictors(i,data,start,end,h,lags)
pred<-Y[(length(Y)-h+1):(length(Y))]
X.tr<-cbind(X$X[(lags+1):(length(Y)-h),],X$Y[(lags+1):(length(Y)-h),1:lags])
glm0<-cv.glmnet(X.tr,log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean)
for (H in 1:h){
X.tr<-cbind(X$X[(lags+1):(length(Y)-h),],X$Y[(lags+1):(length(Y)-h),H:(H+lags-1)])
X.ts<-cbind(X$X[(length(Y)-h+1):(length(Y)),],X$Y[(length(Y)-h+1):(length(Y)),H:(H+lags-1)])
glm<-glmnet(X.tr,log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean,lambda=glm0$lambda.min)
pred[H]<-exp(predict(glm,X.ts)[H]+Y.mean)-1
if (pred[H]> (2*Y.max)) pred[H]<- Y.max
if (pred[H]<0) pred[H]<- 0
}
pred
}

ADL_forecast<-function(data,start,end,h,lags,sku_pre_Dir,get_predictors){
library(glmnet)
library(data.table)
results<-do.call(rbind.data.frame,lapply(1:nrow(data$unit),sku_pre_Dir,data,start,end,h,lags,get_predictors))
results
}

cl <- makeCluster(getOption("cl.cores", length(IRI_storeitem_datasets)))
ADL_1_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,ADL_forecast,7,54,7,1,sku_pre_Dir,get_predictors)
ADL_3_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,ADL_forecast,7,54,7,3,sku_pre_Dir,get_predictors)
stopCluster(cl)

save(ADL_1_forecast_resutls,file='./base forecasters/ADL_1_forecast_resutls')
save(ADL_3_forecast_resutls,file='./base forecasters/ADL_3_forecast_resutls')
##################

################  ARX forecasts
arimaX_forecast<-function(data,start,end,h,lags,get_predictors){
library(forecast)
library(data.table)
results<-matrix(0,nrow(data$unit),h)
for( i in 1:nrow(data$unit)){

Y<-unlist(data$unit[i,start:ncol(data$unit),with=FALSE])
Y.mean=mean(log(Y[1:(length(Y)-h)]+1))
X<-get_predictors(i,data,start,end,h,lags)
X.tr<-X$X[(lags+1):(length(Y)-h),]
coln<-which(unlist(lapply(data.table(X.tr),sd))<0.1)
X.ts<-X$X[(length(Y)-h+1):(length(Y)),]

if(length(coln)>0){
 arimaX<-try(auto.arima(y=log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean,xreg=X.tr[,-coln]),silent=TRUE)
 if('try-error' %in% class(arimaX)){
 arimaX<-try(auto.arima(y=log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean))
 pred<-exp(forecast(arimaX,h)$mean+Y.mean)-1
 }else{
 pred<-exp(forecast(arimaX,xreg=X.ts[,-coln],h)$mean+Y.mean)-1
 }
 }else{
 arimaX<-try(auto.arima(y=log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean,xreg=X.tr))
 if('try-error' %in% class(arimaX)){
 arimaX<-try(auto.arima(y=log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean))
 pred<-exp(forecast(arimaX,h)$mean+Y.mean)-1
 }else{
 pred<-exp(forecast(arimaX,xreg=X.ts,h)$mean+Y.mean)-1
 }
}

 Y.max<-max(Y)
 pred[pred>(2*Y.max)]<- Y.max
 pred[pred<0]<- 0
 results[i,]<- pred 
 
}
results
}

cl <- makeCluster(getOption("cl.cores", length(IRI_storeitem_datasets)))
arimaX_1_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,arimaX_forecast,7,54,7,1,get_predictors)
arimaX_3_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,arimaX_forecast,7,54,7,3,get_predictors)
stopCluster(cl)

save(arimaX_1_forecast_resutls,file='./base forecasters/arimaX_1_forecast_resutls')
save(arimaX_3_forecast_resutls,file='./base forecasters/arimaX_3_forecast_resutls')

#######################################  ELM forecasts

ELM_forecast<-function(data,start,end,h,lags,nhid,get_predictors){
library(elmNNRcpp)
library(data.table)
results<-matrix(0,nrow(data$unit),h)
for( i in 1:nrow(data$unit)){

Y<-unlist(data$unit[i,start:ncol(data$unit),with=FALSE])
Y.max<-max(Y)
Y.mean=mean(log(Y[1:(length(Y)-h)]+1))
Y.tr<-matrix(log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean,length((lags+1):(length(Y)-h)),1)
X<-get_predictors(i,data,start,end,h,lags)
for (H in 1:h){
X.tr<-as.matrix(cbind(X$X[(lags+1):(length(Y)-h),],X$Y[(lags+1):(length(Y)-h),H:(H+lags-1)]))
X.ts<-as.matrix(cbind(X$X[(length(Y)-h+1):(length(Y)),],X$Y[(length(Y)-h+1):(length(Y)),H:(H+lags-1)]))
elmX<-elm_train(X.tr,Y.tr,nhid = nhid,actfun='purelin')
 pred<-exp( elm_predict(elmX,X.ts)[H]+Y.mean)-1
 pred[pred>(2*Y.max)]<- Y.max
 pred[pred<0]<- 0
 results[i,H]<- pred 
}

}
results
}

cl <- makeCluster(getOption("cl.cores", length(IRI_storeitem_datasets)))
ELM_1_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,ELM_forecast,7,54,7,1,5,get_predictors)
ELM_3_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,ELM_forecast,7,54,7,3,5,get_predictors)
stopCluster(cl)

save(ELM_1_forecast_resutls,file='./base forecasters/ELM_1_forecast_resutls')
save(ELM_3_forecast_resutls,file='./base forecasters/ELM_3_forecast_resutls')

######################  SVM forecasts

SVM_forecast<-function(data,start,end,h,lags,get_predictors){
library(e1071)
library(data.table)
results<-matrix(0,nrow(data$unit),h)
for( i in 1:nrow(data$unit)){

Y<-unlist(data$unit[i,start:ncol(data$unit),with=FALSE])
Y.max<-max(Y)
Y.mean=mean(log(Y[1:(length(Y)-h)]+1))
Y.tr<-matrix(log(Y[(lags+1):(length(Y)-h)]+1)-Y.mean,length((lags+1):(length(Y)-h)),1)
X<-get_predictors(i,data,start,end,h,lags)
for (H in 1:h){
X.tr<-as.matrix(cbind(X$X[(lags+1):(length(Y)-h),],X$Y[(lags+1):(length(Y)-h),H:(H+lags-1)]))
X.ts<-as.matrix(cbind(X$X[(length(Y)-h+1):(length(Y)),],X$Y[(length(Y)-h+1):(length(Y)),H:(H+lags-1)]))
svmX<-try(svm(X.tr,Y.tr),silent=TRUE)
 if('try-error' %in% class(svmX)){
 pred<-Y[length(Y)-h]
 }else{ 
pred<-exp( predict(svmX,X.ts)[H]+Y.mean)-1
}
 pred[pred>(2*Y.max)]<- Y.max
 pred[pred<0]<- 0
 results[i,H]<- pred 
}

}
results
}

cl <- makeCluster(getOption("cl.cores", length(IRI_storeitem_datasets)))
SVM_1_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,SVM_forecast,7,54,7,1,get_predictors)
SVM_3_forecast_resutls<-parLapply(cl,IRI_storeitem_datasets,SVM_forecast,7,54,7,3,get_predictors)
stopCluster(cl)

save(SVM_1_forecast_resutls,file='./base forecasters/SVM_1_forecast_resutls')
save(SVM_3_forecast_resutls,file='./base forecasters/SVM_3_forecast_resutls')

####################
