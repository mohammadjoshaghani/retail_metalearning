


categorys<-c("milk","beer","mayo","yogurt","coffee","laundet")  ### The category names to be extracted 
#############################
library(reshape2)
library(data.table)
stores<-fread('Z:/retail dataset/IRI/Academic Dataset External/Year7/External/beer/Delivery_Stores',fill=TRUE)

set.seed(10)
stores<-sample(stores$IRI_KEY,100)  ### sample 100 stores
SKU.data.na<-list()
SKU.data<-list()
Num.SKU<-c()
ncate<-0;
features<-9                           
storedata<-c()

for (catename in categorys){  

ncate<-ncate+1;

year.adj1<-0;year.adj2<-0
for (year in c(5,6,7)){                        

if (year==6) year.adj2<-1;
if (year==7) {year.adj1<-1;year.adj2<-1}

data.file<-paste0("Z:/retail dataset/IRI/Academic Dataset External/","Year",year,"/External/",catename,                ##### need to set your own path to the directory of IRI data
                           "/",catename,"_groc_",1062+year*52+year.adj1,"_",1113+year*52+year.adj2)  
						   
newdata<-fread(file=data.file)
newdata[,cate:=catename]
storedata<-rbind(storedata,newdata[IRI_KEY %in% stores]  )                                                                 ##### read data one year by one year)

}

}

storedata$price<-storedata$DOLLARS/storedata$UNITS
storedata[,F:=ifelse(F=='NONE',0,1)]  
calendar<-fread('time.csv')
names(calendar)[1]<-'WEEK'
names(calendar)[29]<-'fourthJuly'
calendar[,holiday:=(Halloween==1 | Thanksgiving==1 |  Christmas==1 | NewYear==1 | President ==1 | Easter ==1 |
         Memorial==1 | Labour==1 |fourthJuly==1 )]
calendar[is.na(calendar)]<-0
storedata<-calendar[storedata,on='WEEK']
units<-dcast(storedata,IRI_KEY+SY+GE+VEND+ITEM+cate~WEEK, value.var='UNITS',fun.aggregate=mean)
prices<-dcast(storedata,IRI_KEY+SY+GE+VEND+ITEM+cate~WEEK, value.var='price',fun.aggregate=mean)
F<-dcast(storedata,IRI_KEY+SY+GE+VEND+ITEM+cate~WEEK, value.var='F',fun.aggregate=mean)
D<-dcast(storedata,IRI_KEY+SY+GE+VEND+ITEM+cate~WEEK, value.var='D',fun.aggregate=mean)
holiday<-dcast(storedata,IRI_KEY+SY+GE+VEND+ITEM+cate~WEEK, value.var='holiday',fun.aggregate=mean)
w1<-seq(ncol(units),61, by= -7)
w0<-w1-55



data_rolling<-function(header,start,end){
width<-start:end
wa<-units[,c(header,width),with=FALSE]
index<- rowSums(is.na(wa))==0
list(unit=wa[index,],price=prices[index,c(header,width),with=FALSE],
F=F[index,c(header,width),with=FALSE],D=D[index,c(header,width),with=FALSE],
holiday=holiday[index,c(header,width),with=FALSE] )
}

IRI_storeitem_datasets<-list()
header<-1:6
for (i in 1:15){

IRI_storeitem_datasets[[i]]<-data_rolling(header,w0[i]+1,w1[i])

}

save(IRI_storeitem_datasets,file='IRI_storeitem_datasets')



