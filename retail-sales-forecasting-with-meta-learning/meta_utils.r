
get_predictions<-function(pool_type){
load('./base forecasters/ETS_forecast_resutls')
load('./base forecasters/SVM_1_forecast_resutls')
load('./base forecasters/SVM_3_forecast_resutls')
load('./base forecasters/ELM_1_forecast_resutls')
load('./base forecasters/ELM_3_forecast_resutls')
load('./base forecasters/arimaX_1_forecast_resutls')
load('./base forecasters/arimaX_3_forecast_resutls')
load('./base forecasters/ADL_1_forecast_resutls')
load('./base forecasters/ADL_3_forecast_resutls')

load(file='./base forecasters/Xgboost_forecast_7')
load(file='./base forecasters/Xgboost_forecast_3')
load(file='./base forecasters/Lasso_Dir_7')
load(file='./base forecasters/Lasso_Dir_3')
load(file='./base forecasters/Ranger_Dir_7')
load(file='./base forecasters/Ranger_Dir_3')
load(file='./base forecasters/ELM_pool_Dir_7')
load(file='./base forecasters/ELM_pool_Dir_3')


models_individual<-list(ETS_f=ETS_forecast_resutls,  
ADL_1=ADL_1_forecast_resutls,ADL_3=ADL_3_forecast_resutls,
arimaX_1=arimaX_1_forecast_resutls,arimaX_3=arimaX_3_forecast_resutls,
SVM_1=SVM_1_forecast_resutls,SVM_3=SVM_3_forecast_resutls,
ELM_1=ELM_1_forecast_resutls, ELM_3=ELM_3_forecast_resutls)

models_pool<-list(Xgb3=Xgboost_forecast_3,Xgb7=Xgboost_forecast_7,
Lasso3=Lasso_Dir_3,Lasso7=Lasso_Dir_7,
RF3=Ranger_Dir_3,RF7=Ranger_Dir_7,
Elmp3=ELM_pool_Dir_3,Elmp7=ELM_pool_Dir_7)

models_mixed<-list(ETS_f=ETS_forecast_resutls, ADL_1=ADL_1_forecast_resutls,
arimaX_1=arimaX_1_forecast_resutls, ELM_1=ELM_1_forecast_resutls, 
SVM_1=SVM_1_forecast_resutls, Xgb=Xgboost_forecast_7,Lasso3=Lasso_Dir_3,
ranger=Ranger_Dir_7,elmp=ELM_pool_Dir_7)

if (pool_type=='individual') pred= models_individual
if (pool_type=='pooling') pred=models_pool
if (pool_type=='mixed') pred=models_mixed
pred
}



base_forecasts_metalearning_prepare <-function(base_forecasts,Y_train,slot_tr,slot_ts){

### split the forecasts of a base-forecaster into training and test periods
M_pred<-function(model){
train<-c() ;  test<-c()

for (i in slot_tr){
a<-as.matrix(model[[i]])
colnames(a)<-NULL
train<-rbind(train,a)
}
for (i in slot_ts){
a<-as.matrix(model[[i]])
colnames(a)<-NULL
test<-rbind(test,a)
}
list(train=train,test=test)
}

### reshape the forecasts into 3d array

reshape_pred_3d <-function(pred,set){
nmodels<-length(pred)
nsample<-dim(pred[[1]][set][[1]])[1]
h<-dim(pred[[1]][set][[1]])[2]
Y <- array(0, dim = c(nsample,nmodels,h))

for (i in 1:nmodels){
a<-pred[[i]][set][[1]]
a[a==Inf]<-mean(a[a!=Inf])
a[a<0]<-0
Y[,i,]<-a

}
Y
}

##  main process
models_pred<-lapply(base_forecasts, M_pred)
pred_train<-reshape_pred_3d (models_pred,'train')
pred_test<-reshape_pred_3d (models_pred,'test')

### bias adj for base forecasters
for (m in 1:length(base_forecasts)) {
pred<-data.table(pred_train[,m,])
real<-data.table(Y_train)
bias.adj<-nrow(real)/sum(rowSums(pred)/rowSums(real))
pred_train[,m,]<-pred_train[,m,]*bias.adj
pred_test[,m,]<-pred_test[,m,]*bias.adj
}

list(pred_train,pred_test)

}


get_features<-function(X){
library(tsfeatures)
X_t<-ts(t(X))
X_f<-tsfeatures(X_t,features = c("acf_features", 
                "arch_stat", "crossing_points", "entropy", "flat_spots", 
                 "holt_parameters", "hurst", "lumpiness", "nonlinearity", "pacf_features", 
                "stl_features", "stability", "unitroot_kpss", "unitroot_pp"),parallel=TRUE,scale=FALSE)
X_f[is.na(X_f)]	<-0		
X_f	
}

get_features_P<-function(X){
library(tsfeatures)
X_t<-ts(t(X))
X_f<-tsfeatures(X_t,features = c("crossing_points", "entropy", "flat_spots", 
                  "hurst", "lumpiness", "stability"),parallel=TRUE,scale=FALSE)
X_f[is.na(X_f)]	<-0		
X_f	
}

reshape_X_3d<-function(X,nf){
k<-ncol(X)/nf
Y <- array(0, dim = c(nrow(X),k,nf))
for (i in 1:nf) Y[,,i]<-as.matrix((X[,((i-1)*k+1):(i*k)]))
Y
}

na_fill_zero<-function (x) {
x[is.na(x)]<-0
x
}




acc_best<-function(pred,real){
models<-dim(pred)[2]
acc_mse<-matrix(0,dim(pred)[1],dim(pred)[2])
for (i in 1:models) acc_mse[,i]<-rowMeans((pred[,i,]-real)^2)
acc_min<-apply(acc_mse,1,min)
acc_one<-acc_mse==acc_min
acc_one<-acc_one/apply(acc_one,1,sum)
acc_one
}

acc_err<-function(pred,real){
models<-dim(pred)[2]
acc_mse<-matrix(0,dim(pred)[1],dim(pred)[2])
for (i in 1:models) acc_mse[,i]<-rowMeans(abs(pred[,i,]-real))
acc_mse
}



data_metalearning_prepare<-function(st,horizon,slot_tr,slot_ts,data){

X_train_y<-c()
X_train_P<-c()
X_test_y<-c()
X_test_P<-c()
Y_train<-c()
Y_test<-c()


for (i in slot_tr){
X_train_y<- rbind(X_train_y, data[[i]]$unit[,st[1]:st[2]]/apply(data[[i]]$unit[,st[1]:st[2]],1,mean),use.names=FALSE)
X_train_P<- rbind(X_train_P,
cbind(
data[[i]]$F[,st[1]:(st[2]+horizon)],  ## advertising series
data[[i]]$D[,st[1]:(st[2]+horizon)],  ## display series
data[[i]]$price[,(st[1]):(st[2]+horizon)]/apply(data[[i]]$price[,st[1]:st[2]],1,mean)  ### decaled price series
),use.names=FALSE)
Y_train<- rbind(Y_train,data[[i]]$unit[,(st[2]+1):(st[2]+horizon)],use.names=FALSE)  ## true sales in training periods
}


for (i in slot_ts){
X_test_y<- rbind(X_test_y, data[[i]]$unit[,st[1]:st[2]]/apply(data[[i]]$unit[,st[1]:st[2]],1,mean), use.names=FALSE)
X_test_P<- rbind(X_test_P,
cbind(
data[[i]]$F[,st[1]:(st[2]+horizon)],
data[[i]]$D[,st[1]:(st[2]+horizon)],
data[[i]]$price[,(st[1]):(st[2]+horizon)]/apply(data[[i]]$price[,st[1]:st[2]],1,mean)
),use.names=FALSE)
Y_test<- rbind(Y_test,data[[i]]$unit[,(st[2]+1):(st[2]+horizon)],use.names=FALSE) ## true sales in training periods
}


colnames(X_train_y)<-as.character(1:ncol(X_train_y))
colnames(X_train_P)<-as.character(1:ncol(X_train_P))


multi_y_trained<- recipe(~.,X_train_y) %>%  step_center(all_predictors()) %>%  step_scale(all_predictors())%>% prep()
multi_P_trained<- recipe(~.,X_train_P) %>%  step_center(all_predictors()) %>%  step_scale(all_predictors())%>% prep()

X_train_y<- as.matrix(bake(multi_y_trained, X_train_y)) %>%  na_fill_zero()  %>% reshape_X_3d (1)
X_train_P<- as.matrix(bake(multi_P_trained, X_train_P)) %>%  na_fill_zero()  %>% reshape_X_3d (3)

colnames(X_test_y)<-as.character(1:ncol(X_test_y))
colnames(X_test_P)<-as.character(1:ncol(X_test_P))

X_test_y<- as.matrix(bake(multi_y_trained,  X_test_y))  %>%  na_fill_zero()  %>% reshape_X_3d (1)
X_test_P<- as.matrix(bake(multi_P_trained,  X_test_P))  %>%  na_fill_zero()  %>% reshape_X_3d (3)

list(as.matrix(Y_train),as.matrix(Y_test),X_train_y,X_train_P,X_test_y,X_test_P) 
}

#### used to calculated scale fator in loss function

get_scaler<-function(forecasts,real){
mds<-dim(forecasts)[2]
scaler<-matrix(0,nrow(real),mds)
for (i in 1:mds) scaler[,i]<-rowMeans((forecasts[,i,]-real)^2)
rowMeans(scaler)
}



acc_h<-function(pred,real,base,bias.adj){
pred<-pred*bias.adj
base<-base*bias.adj
g_mean<-function(x){exp(mean(log(x[is.finite(log(x))]))) }
MAE<-(colMeans(abs(real-pred),na.rm=TRUE))
RMSE<-(colMeans((real-pred)^2,na.rm=TRUE)^0.5)
sMAPE<-(colMeans(abs(real-pred)/abs(real+pred),na.rm=TRUE)*100)
sMdAPE<-(matrixStats::colMedians(as.matrix(abs(real-pred)/abs(real+pred)),na.rm=TRUE)*100)
arelmae= (apply(abs(pred-real)/abs(base-real),2,g_mean))
rbind(sMAPE,sMdAPE,arelmae)
}


acc_all<-function(pred,real,base,bias.adj){
pred<-pred*bias.adj
base<-base*bias.adj
g_mean<-function(x){exp(mean(log(x[is.finite(log(x))]))) }
ME<-mean(rowSums(pred-real,na.rm=TRUE)/rowSums(real,na.rm=TRUE)*100,na.rm=TRUE)
MAE<-mean(colMeans(abs(real-pred),na.rm=TRUE),na.rm=TRUE)
RMSE<-mean(colMeans((real-pred)^2,na.rm=TRUE)^0.5,na.rm=TRUE)
sMAPE<-mean(rowMeans(abs(real-pred)/abs(real+pred),na.rm=TRUE)*100,na.rm=TRUE)
sMdAPE<-mean(matrixStats::rowMedians(as.matrix(abs(real-pred)/abs(real+pred)),na.rm=TRUE)*100,na.rm=TRUE)
arelmae= g_mean(rowMeans(abs(pred-real),na.rm=TRUE)/rowMeans(abs(base-real),na.rm=TRUE))
cbind(sMAPE,sMdAPE,arelmae,ME)
}
