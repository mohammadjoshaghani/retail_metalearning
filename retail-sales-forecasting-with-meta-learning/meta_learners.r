


scale_mse <- function( scales ) {    ### loss function 
loss<-function(y_true, y_pred){
	k_mean( (k_mean(k_square(y_pred-y_true),axis=-1,keepdims=TRUE)/scales))
   }
loss
}


fcn_block<-function(input){     ### convonlutional blocks

squeeze_excite_block<-function(input){
    filters = unlist(k_get_variable_shape(input)[3])
    se = layer_global_average_pooling_1d(input) %>%
        layer_dense(filters %/% 16,  activation='relu', kernel_initializer='he_normal', use_bias=FALSE)%>%
        layer_dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=FALSE)
    se = layer_multiply(list(input, se))
	se
}

fcn<-input %>%
  layer_conv_1d(filters = 64, kernel_size = 2, padding="same", kernel_initializer='he_uniform') %>%
 layer_activation_relu() %>%
  squeeze_excite_block() %>%
  layer_conv_1d(filters = 128, kernel_size = 4, padding="same", kernel_initializer='he_uniform') %>%
  layer_activation_relu() %>%
  squeeze_excite_block() %>%
  layer_conv_1d(filters = 64, kernel_size = 8, padding="same", kernel_initializer='he_uniform') %>%
 layer_activation_relu() %>%
 layer_global_average_pooling_1d()
 fcn
 }
 

M0<-function(ep=50,XX=TRUE,y_shape,P_shape,mds,horizons,train_input,Y_train,test_input,Y_test){   
## this meta learner is used to train M0, M2, M3,and M4, depending on the data inputed and value of XX
## ep: number of epochs of training;
## XX: whether extracting features from indicators
## y_shape: shape of sales series
## P_shape: shape of indicator series
## mds: number of base forecasters
## horizons: forecasting horizon

X_input_y <-layer_input(shape = y_shape)
X_input_P <-layer_input(shape = P_shape)
L_input   <-  layer_input(shape = c(1))  
y_input<-layer_input(shape = c(mds,horizons))

X_layers_y <- X_input_y  %>%  fcn_block()
X_layers_P <- X_input_P  %>%  fcn_block()

if (XX) {
X_layers <- layer_concatenate(list(X_layers_y,X_layers_P)) %>%layer_dropout(0.8)%>% layer_dense(units = mds,activation="softmax" ) %>%
layer_repeat_vector(horizons) } else {
X_layers <- X_layers_y %>%layer_dropout(0.8)%>% layer_dense(units = mds,activation="softmax" ) %>%
layer_repeat_vector(horizons) }
 
y_input_p<- y_input %>% layer_permute(list(2,1))

multi_sum<-function(args) k_sum(args,axis=-1) 
output <-layer_multiply(input=list(y_input_p,X_layers)) %>%  layer_lambda(multi_sum)

model <- keras_model(list(X_input_y,X_input_P,y_input,L_input),output)

model %>%  compile( loss = scale_mse(L_input), optimizer = 'adam' )


history <- model %>% fit(
  train_input,Y_train, 
  validation_data = list(test_input, Y_test),
  epochs = ep, batch_size = 4096)
  
model   ## the trained model is returned
}



M1<-function(ep=50,y_shape,mds,horizons,train_input,Y_train,test_input,Y_test){
######## meta learner using hand craft features

X_input <-layer_input(shape = y_shape)
L_input   <-  layer_input(shape = c(1))
y_input<-layer_input(shape = c(mds,horizons))

X_layers<-X_input %>% layer_dense(128,activation='relu') %>%
layer_dense(64,activation='relu') %>%
layer_dense(32,activation='relu') %>%
layer_dropout(0.8)%>%
layer_dense(units = mds,activation="softmax" ) %>%
layer_repeat_vector(horizons) 
 
y_input_p<- y_input %>% layer_permute(list(2,1))

multi_sum<-function(args) k_sum(args,axis=-1) 
output <-layer_multiply(input=list(y_input_p,X_layers)) %>%  layer_lambda(multi_sum)

model <- keras_model(list(X_input,y_input,L_input),output)

model %>%  compile( loss = scale_mse(L_input), optimizer = 'adam' )

history <- model %>% fit(
  train_input,Y_train, 
  validation_data = list(test_input, Y_test),
  epochs = ep, batch_size = 4096)
  
 model
  }
  
 
M5<-function(ep=50,y_shape,mds,train_input,Y_train_bestone,test_input,Y_test_bestone){ 


X_input_y <-layer_input(shape = y_shape)
X_input_P <-layer_input(shape = P_shape) 

X_layers_y <- X_input_y  %>%  fcn_block()
X_layers_P <- X_input_P  %>%  fcn_block()

X_layers <- layer_concatenate(list(X_layers_y,X_layers_P)) %>%layer_dropout(0.8)%>%
  layer_dense(units = mds,activation="softmax" ) 
model <- keras_model(list(X_input_y,X_input_P), X_layers)

model %>%
  compile(
   loss ='categorical_crossentropy',
    optimizer = 'adam',
	metrics = c("accuracy")
  )
  
history <- model %>% fit(
  train_input,Y_train_bestone, 
  validation_data = list(test_input, Y_test_bestone),
  epochs = ep, batch_size = 4096)
  
model
  }
  
  
  
M6<-function(ep=50,y_shape,P_shape,mds,train_input,Y_train_err,test_input,Y_test_err){   

X_input_y <-layer_input(shape = y_shape)
X_input_P <-layer_input(shape = P_shape) 

X_layers_y <- X_input_y  %>%  fcn_block()
X_layers_P <- X_input_P  %>%  fcn_block()

X_layers <- layer_concatenate(list(X_layers_y,X_layers_P)) %>%layer_dropout(0.8)%>%
  layer_dense(units = mds,activation='relu' ) 
model <- keras_model(list(X_input_y,X_input_P), X_layers)

model %>%
  compile(
   loss =loss_mean_squared_error, optimizer = 'adam'
  )

history <- model %>% fit(
  train_input,Y_train_err, 
  validation_data = list(test_input, Y_test_err),
  epochs = ep, batch_size = 4096)
  
model  
}


FFORMA<-function(ep=100,X_train,Y_train_err,X_test,Y_test_err){ 

 dtrain <- xgboost::xgb.DMatrix(X_train,label=apply(Y_train_err,1,function(x) which(x==min(x))-1))
 attr(dtrain, "errors") <- Y_train_err
  dtest<- xgboost::xgb.DMatrix(X_test,label=apply(Y_test_err,1,function(x) which(x==min(x))-1))
 attr(dtest, "errors") <- Y_test_err
 
        param <- list(max_depth = 14, eta = 0.575188, nthread = 3, 
            silent = 0, objective = error_softmax_obj, num_class = ncol(Y_train_err), 
            subsample = 0.9161483, colsample_bytree = 0.7670739)	
			
 watchlist <- list(train = dtrain,valid=dtest)
  model <- xgboost::xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = ep,
                    verbose             = 1,
                    watchlist           = watchlist, 
                    maximize            = FALSE
)

model
}






  
  

