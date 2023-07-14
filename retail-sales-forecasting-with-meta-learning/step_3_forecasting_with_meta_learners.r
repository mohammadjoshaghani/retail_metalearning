


library(reshape2)
library(data.table)
library(recipes)
library(keras)
source('meta_utils.r')
source('meta_learners.r')

###  training and generate forecasts for M0, M2,M3, and M4 
slot_tr=c(6:15)
slot_ts=c(1:5)

load('IRI_storeitem_datasets')
data=IRI_storeitem_datasets

#### split the data into training and test set, and data pooling
c(Y_train,Y_test,X_train_y,X_train_P,X_test_y,X_test_P)  %<-% data_metalearning_prepare(st=c(7,54),horizon=7,slot_tr,slot_ts,data=data)

### load base forecasts and split them into training and test set
base_forecasts<- get_predictions('mixed')  ## 'individual" for M3, "pooling" for M4, "mixed" for M0 and M2
c(base_forecasts_train,base_forecasts_test) %<-% base_forecasts_metalearning_prepare (base_forecasts,Y_train,slot_tr,slot_ts)


L_train<-get_scaler(base_forecasts_train,Y_train)
L_test<-get_scaler(base_forecasts_test,Y_test)

horizons<-dim(base_forecasts_train)[3]
mds<-dim(base_forecasts_test)[2]
y_shape=c(dim(X_train_y)[2:3])
P_shape=c(dim(X_train_P)[2:3])



train_input<-list(X_train_y,X_train_P,base_forecasts_train,L_train)  ### train inputs to meta learners
test_input<-list(X_test_y,X_test_P, base_forecasts_test,L_test)  ### test inputs to meta learners

nn<-10  # times of predictions to be averaged, the larger number the more stable forecasts
predictions<-array(0,c(dim(Y_test),nn))

for (i in 1:nn)   {
model<-M0(ep=50,XX=TRUE,y_shape,P_shape,mds,horizons,train_input,Y_train,test_input,Y_test) 
predictions[,,i]<- model %>%  predict(test_input)

}

M0_forecasts<-Y_test 
for (i in 1:horizons) M0_forecasts[,i]<-rowMeans(predictions[,i,])

acc_h(M0_forecasts,Y_test,base_forecasts_test[,6,],bias.adj=1)
acc_all(M0_forecasts,Y_test,base_forecasts_test[,6,],bias.adj=1)

save(M0_forecasts,file='M0_forecasts.dat')

################# training M6

Y_train_err<-acc_err(base_forecasts_train,Y_train)
Y_test_err<-acc_err(base_forecasts_test,Y_test)

train_input<-list(X_train_y,X_train_P)
test_input<-list(X_test_y,X_test_P)

predictions<-array(0,c(dim(base_forecasts_test)[1:2],nn))
for (i in 1:nn)   {
model<-M6(ep=50,y_shape,P_shape,mds,train_input,Y_train_err,test_input,Y_test_err)
predictions[,,i]<- model %>%  predict(test_input)
}

M_forecasts<-base_forecasts_test[,,1] 
for (i in 1:mds) M_forecasts[,i]<-rowMeans(predictions[,i,])


pred_weight<-t(apply(M_forecasts,1, function(x){exp(-x)/sum(exp(-x))} ))

M6_forecasts<-c()
for (i in 1:horizon) { M6_forecasts<-cbind( M6_forecasts,rowSums(pred_weight*base_forecasts_test[,,i])) }

############## training M5


Y_train_bestone<-acc_best(base_forecasts_train,Y_train)
Y_test_bestone<-acc_best(base_forecasts_test,Y_test)
 
train_input<-list(X_train_y,X_train_P)
test_input<-list(X_test_y,X_test_P)

predictions<-array(0,c(dim(base_forecasts_test)[1:2],nn))
for (i in 1:nn)   {
model<-M5(ep=50,y_shape,mds,train_input,Y_train_bestone,test_input,Y_test_bestone)
predictions[,,i]<- model %>%  predict(test_input)
}
M_forecasts<-base_forecasts_test[,,1] 
for (i in 1:mds) M_forecasts[,i]<-rowMeans(predictions[,i,])

pred_best<-apply(M_forecasts,1, function(x){ which(x==max(x))[1]} )

M5_forecasts<-matrix(0,dim(base_forecasts_test)[1],dim(base_forecasts_test)[3])
for (i in 1:dim(base_forecasts_test)[1] ){
M5_forecasts[i,]<-base_forecasts_test[i,pred_best[i],]
}


acc_h(M5_forecasts,Y_test,base_forecasts_test[,6,],bias.adj=1)
acc_all(M5_forecasts,Y_test,base_forecasts_test[,6,],bias.adj=1)

################## Training M1 with hand selected features

X_train_Y_features<-get_features(X_train_y[,,1])
uselessCols<-which(unlist(lapply(X_train_Y_features,var))==0)
X_train_Y_features<-X_train_Y_features[,-uselessCols]

X_test_Y_features<-get_features(X_test_y[,,1])
X_test_Y_features<-X_test_Y_features[,-uselessCols]

X_train_P_features<-get_features_P(X_train_P[,,3])  ## price
X_train_D_features<-get_features_P(X_train_P[,,2])  ## display
X_train_F_features<-get_features_P(X_train_P[,,1])  ## feature advertising

X_test_P_features<-get_features_P(X_test_P[,,3])
X_test_D_features<-get_features_P(X_test_P[,,2])
X_test_F_features<-get_features_P(X_test_P[,,1])


X_train_Y_features<-cbind(X_train_Y_features,X_train_P_features,X_train_D_features, X_train_F_features)
names(X_train_Y_features)<-1:ncol(X_train_Y_features)
X_test_Y_features<-cbind(X_test_Y_features,X_test_P_features,X_test_D_features, X_test_F_features)
names(X_test_Y_features)<-1:ncol(X_test_Y_features)

X_recipe<- recipe(~.,X_train_Y_features) %>%    step_center(all_predictors()) %>%    step_scale(all_predictors())%>%    prep()
X_train_Y<- as.matrix(bake(X_recipe, X_train_Y_features))
X_test_Y<- as.matrix(bake(X_recipe,  X_test_Y_features))


y_shape<-dim(X_train_Y)[2]
  
train_input<-list(X_train_Y,base_forecasts_train,L_train)
test_input<-list(X_test_Y, base_forecasts_test,L_test)

predictions<-array(0,c(dim(Y_test),nn))
for (i in 1:nn)   {
predictions[,,i]<-M1(ep=50,y_shape,mds,horizons,train_input,Y_train,test_input,Y_test) %>%  predict(test_input)
}

M1_forecasts<-Y_test 
for (i in 1:horizons) M1_forecasts[,i]<-rowMeans(predictions[,i,])

############### FFORMA

Y_train_err<-acc_scaled_err(base_forecasts_train,Y_train,L_train)
Y_test_err<-acc_scaled_err(base_forecasts_test,Y_test,L_test)
library(M4metalearning)
 dtrain <- xgboost::xgb.DMatrix(X_train_Y,label=apply(Y_train_err,1,function(x) which(x==min(x))-1))
 attr(dtrain, "errors") <- Y_train_err
  dtest<- xgboost::xgb.DMatrix(X_test_Y,label=apply(Y_test_err,1,function(x) which(x==min(x))-1))
 attr(dtest, "errors") <- Y_test_err
 
        param <- list(max_depth = 14, eta = 0.575188, nthread = 3, 
            silent = 0, objective = error_softmax_obj, num_class = ncol(Y_train_err), 
            subsample = 0.9161483, colsample_bytree = 0.7670739)	
			
 watchlist <- list(train = dtrain,valid=dtest)
  bst <- xgboost::xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 200,
                    verbose             = 1,
                    watchlist           = watchlist, 
                    maximize            = FALSE
)

  pred <- stats::predict(bst, X_test_Y, outputmargin = TRUE, reshape = TRUE)
  
  
  pred <- t(apply(pred, 1, softmax_transform))
  
  FFORMA_f<-c()
  for (h in 1:dim(base_forecasts_test)[3]){
  FFORMA_f<- cbind(FFORMA_f,rowSums(pred * base_forecasts_test[,,h]))
  }
  
acc_h(FFORMA_f,Y_test,base_forecasts_test[,6,],bias.adj=1)
acc_all(FFORMA_f,Y_test,base_forecasts_test[,6,],bias.adj=1)


############ simple ensemble

E1<-cbind(rowMeans(base_forecasts_test[,,1]),
rowMeans(base_forecasts_test[,,2]),
rowMeans(base_forecasts_test[,,3]),
rowMeans(base_forecasts_test[,,4]),
rowMeans(base_forecasts_test[,,5]),
rowMeans(base_forecasts_test[,,6]),
rowMeans(base_forecasts_test[,,7]) )


Y_train_err<-acc_err(base_forecasts_train,Y_train)
pred_weight<-exp(-colMeans((Y_train_err)))/sum(exp(-colMeans((Y_train_err))))

E2<-cbind(rowSums(t(pred_weight*t(base_forecasts_test[,,1]))),
rowSums(t(pred_weight*t(base_forecasts_test[,,2]))),
rowSums(t(pred_weight*t(base_forecasts_test[,,3]))),
rowSums(t(pred_weight*t(base_forecasts_test[,,4]))),
rowSums(t(pred_weight*t(base_forecasts_test[,,5]))),
rowSums(t(pred_weight*t(base_forecasts_test[,,6]))),
rowSums(t(pred_weight*t(base_forecasts_test[,,7]))) )

base_forecasts<- get_predictions('pooling') 
c(base_forecasts_train,base_forecasts_test) %<-% base_forecasts_metalearning_prepare (base_forecasts,Y_train,slot_tr,slot_ts)
selected<- c(1,2,5,6) ###best 4,including ranger 3 &7, xgb 3&7

E3<-cbind(rowMeans(base_forecasts_test[,selected,1]),
rowMeans(base_forecasts_test[,selected,2]),
rowMeans(base_forecasts_test[,selected,3]),
rowMeans(base_forecasts_test[,selected,4]),
rowMeans(base_forecasts_test[,selected,5]),
rowMeans(base_forecasts_test[,selected,6]),
rowMeans(base_forecasts_test[,selected,7]) )

acc_h(E3,Y_test,base_forecasts_test[,6,],bias.adj=1)
acc_all(E3,Y_test,base_forecasts_test[,6,],bias.adj=1)


