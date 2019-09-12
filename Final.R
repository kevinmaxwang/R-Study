# Final Project
# Zihao Xu (912845955) Mingye Yang (912777230) Zhixin Zheng (912903198)

setwd("D:/Michael Xu/Documents/UC Davis/OneDrive - UC Davis/Fall 2017/STA 141A/datasets/Final")
setwd("C:/Users/micha/OneDrive - UC Davis/Fall 2017/STA 141A/datasets/Final")

#1
#Read and check data 
read_digits<-function(p){
  df<-as.data.frame(read.table(p))
  df$V1<-as.factor(df$V1)
  names(df)[1]<-"digit"
  df
}

test<-read_digits("test.txt")
train<-read_digits("train.txt")

#2
#https://stackoverflow.com/questions/22509106/converting-a-matrix-into-a-gray-scale-image-in-r
#https://stackoverflow.com/questions/16496210/rotate-a-matrix-in-r
#display image of designated row 
view_digit<-function(n,df){
  m<-matrix(as.numeric(df[n,c(2:257)]),ncol = 16,byrow=T)
  image(t(m[nrow(m):1,]),col=paste("gray",1:99,sep=""),axes=F)
  df[n,1]
}

#3
#a
#Display what each digit looks like on average  
avr_pixel<-function(n,df){
  avr<-t(matrix(c(n,sapply(df[df$digit==n,c(2:257)], mean))))
  as.data.frame(avr)
  view_digit(1,avr)
}

par(mfrow=c(2,5))
for (i in 0:9) {
  avr_pixel(i,train)
}

#b
#Compute variance of each column 
var_pixel<-function(df){
  var<-as.data.frame(matrix(sapply(df[,c(2:257)], var)))
  names(var)<-"var"
  var
}

train_var<-var_pixel(train)
r=1
train_var<-var_pixel(train)
view_digit(1,t(as.matrix(t(rbind(train_var,r))[c(257,1:256)])))

#order variance and find the lowest and highest  
order(train_var)
train_var[c(230,219,105,185,121),]
train_var[c(241,1,256,16,17),]
train_var[1,]

#4
#Write a KNN function to predict for a point  
predict_knn<-function(p,t,d="euclidean",k){
  x<-nrow(t)
  y<-nrow(p)
  b<-rbind(t,p)
  m<-as.matrix(dist(b[,c(2:257)], method = d))
  for (i in 1:y) {
    v<-data.frame(table(t[order(m[x+i,c(1:x)])[c(1:k)],1]))$Freq
    max<-which(v==max(v))
    p$digit[i]<-as.numeric(sample(as.character(max),1))-1
  }
  p$digit
}

#5
#https://www.youtube.com/watch?v=p5rDg4OVBuA
#Write a function of using 10-fold cross-validation to estimate the error rate for k-nearest neighbors 
cv_error_knn<-function(tt,d="euclidean",k,f=10){
  x<-nrow(tt)
  a<-as.integer(x/f)
  avr<-c(1:10)
  r<-sample(x,replace = F)
  tt<-tt[r,]
  m<-as.matrix(dist(tt[,c(2:257)], method = d))
  for (i in 1:f) {
    p<-tt[c((1+(i-1)*a):(a*i)),]
    t<-tt[-c((1+(i-1)*a):(a*i)),]
    pt<-p[,1]
    z=1
    for (j in (1+(i-1)*a):(a*i)) {
      v<-data.frame(table(t[order(m[j,-c((1+(i-1)*a):(a*i))])[c(1:k)],1]))$Freq
      max<-which(v==max(v))
      p$digit[z]<-as.numeric(sample(as.character(max),1))-1
      z<-z+1
    }
    b<-p$digit
    cm<-table(pt,b)
    avr[i]<-sum(diag(cm)) / a * 100
  }
  100-mean(avr)
}

#6
#3 distance methods to find the error rate by using cross validation  
cv_error_knnl<-function(tt,d="euclidean",k,f=10){
  x<-nrow(tt)
  a<-as.integer(x/f)
  avr<-c(1:10)
  r<-sample(x,replace = F)
  tt<-tt[r,]
  m<-as.matrix(dist(tt[,c(2:257)], method = d))
  er<-c(1:k)
  for (l in 1:k) {
    for (i in 1:f) {
      p<-tt[c((1+(i-1)*a):(a*i)),]
      t<-tt[-c((1+(i-1)*a):(a*i)),]
      pt<-p[,1]
      z=1
      for (j in (1+(i-1)*a):(a*i)) {
        v<-data.frame(table(t[order(m[j,-c((1+(i-1)*a):(a*i))])[c(1:l)],1]))$Freq
        max<-which(v==max(v))
        p$digit[z]<-as.numeric(sample(as.character(max),1))-1
        z<-z+1
      }
      b<-p$digit
      cm<-table(pt,b)
      avr[i]<-sum(diag(cm)) / a * 100
    }
  er[l]<-100-mean(avr)
  }
  er
}

#call the function and use three different methods 
er<-cv_error_knnl(train,k=15)
er_m<-cv_error_knnl(train,d="manhattan",k=15)
er_c<-cv_error_knnl(train,d="canberra",k=15)

#display error rate 
er
er_m
er_c

#plot results into one graph 
plot(er,ylim = c(2.9,6),xlab = "K",ylab = "Error Rate (in percentage)", main = "Error Rate of Different K under 3 Distance Metric (Train)",xaxt='n')
axis(1, at = seq(1, 15, by = 1), las=1) 
#https://stackoverflow.com/questions/11775692/how-to-specify-the-actual-x-axis-values-to-plot-as-x-axis-ticks-in-r
points(er_m,pch = 2)
points(er_c,pch = 3)
lines(er)
lines(er_m,col = "red")
lines(er_c,col = "green")
legend("topleft",legend = c("Euclidean","Manhattan","Canberra"),col = c("black","red","green"),lty = c(1:3),cex = 0.8,title = "Distance Metric Types")

#7
#Function of building three best combinations' confusion matrix 
conf_mat<-function(tt,d="euclidean",k,f=10){
  x<-nrow(tt)
  a<-as.integer(x/f)
  avr<-c(1:10)
  y<-c(0:9)
  cm<-as.matrix(table(y,y))
  cm<-cm-cm
  r<-sample(x,replace = F)
  tt<-tt[r,]
  m<-as.matrix(dist(tt[,c(2:257)], method = d))
  for (i in 1:f) {
    p<-tt[c((1+(i-1)*a):(a*i)),]
    t<-tt[-c((1+(i-1)*a):(a*i)),]
    pt<-as.numeric(p[,1])
    z=1
    for (j in (1+(i-1)*a):(a*i)) {
      v<-data.frame(table(t[order(m[j,-c((1+(i-1)*a):(a*i))])[c(1:k)],1]))$Freq
      max<-which(v==max(v))
      p$digit[z]<-as.numeric(sample(as.character(max),1))-1
      z<-z+1
    }
    b<-as.numeric(p$digit)
    cmt<-as.matrix(prop.table(table(pt,b)))
    cm<-cm+cmt
  }
  cm/10
}

cm_e1_prop<-conf_mat(train,k=1)
cm_e3_prop<-conf_mat(train,k=3)
cm_e4_prop<-conf_mat(train,k=4)

library(rJava)
library(xlsxjars)
library(xlsx)
write.xlsx(cm_e1_prop,"C:/Users/micha/OneDrive - UC Davis/Fall 2017/STA 141A/datasets/Final/cm_e1_prop.xlsx")
write.xlsx(cm_e3_prop,"C:/Users/micha/OneDrive - UC Davis/Fall 2017/STA 141A/datasets/Final/cm_e3_prop.xlsx")
write.xlsx(cm_e4_prop,"C:/Users/micha/OneDrive - UC Davis/Fall 2017/STA 141A/datasets/Final/cm_e4_prop.xlsx")

#8
#using the best combination to find misclassification rates  
conf_mat_misc<-function(tt,d="euclidean",k,f=10){
  x<-nrow(tt)
  a<-as.integer(x/f)
  avr<-c(1:10)
  y<-c(0:9)
  cm<-as.matrix(table(y,y))
  cm<-cm-cm
  r<-sample(x,replace = F)
  tt<-tt[r,]
  m<-as.matrix(dist(tt[,c(2:257)], method = d))
  for (i in 1:f) {
    p<-tt[c((1+(i-1)*a):(a*i)),]
    t<-tt[-c((1+(i-1)*a):(a*i)),]
    pt<-as.numeric(p[,1])
    z=1
    for (j in (1+(i-1)*a):(a*i)) {
      v<-data.frame(table(t[order(m[j,-c((1+(i-1)*a):(a*i))])[c(1:k)],1]))$Freq
      max<-which(v==max(v))
      p$digit[z]<-as.numeric(sample(as.character(max),1))-1
      z<-z+1
    }
    b<-as.numeric(p$digit)
    cmt<-as.matrix(table(pt,b))
    cm<-cm+cmt
  }
  cm
}

cm_e1<-conf_mat_misc(train,k=1)
cm_e1

#Display the result graphically 
par(mfrow=c(2,5))
barplot(cm_e1[c(1:9),10],ylim = c(0,20),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 9's Missclassified Digits")
barplot(cm_e1[c(1:8,10),9],ylim = c(0,5),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 8's Missclassified Digits")
barplot(cm_e1[c(1:7,9:10),8],ylim = c(0,14),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 7's Missclassified Digits")
barplot(cm_e1[c(1:6,8:10),7],ylim = c(0,8),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 6's Missclassified Digits")
barplot(cm_e1[c(1:5,7:10),6],ylim = c(0,5),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 5's Missclassified Digits")
barplot(cm_e1[c(1:4,6:10),5],ylim = c(0,14),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 4's Missclassified Digits")
barplot(cm_e1[c(1:3,5:10),4],ylim = c(0,4),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 3's Missclassified Digits")
barplot(cm_e1[c(1:2,4:10),3],ylim = c(0,5),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 2's Missclassified Digits")
barplot(cm_e1[c(1,3:10),2],ylim = c(0,7),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 1's Missclassified Digits")
barplot(cm_e1[c(2:10),1],ylim = c(0,10),xlab = "Digit",ylab = "Frequency",main = "Predict Digit 0's Missclassified Digits")

#9
#Write a function of calculating the error rate of actual test prediction. 
predict_knnl<-function(p,t,d="euclidean",k){
  x<-nrow(t)
  y<-nrow(p)
  b<-rbind(t,p)
  pt<-p[,1]
  er<-c(1:k)
  m<-as.matrix(dist(b[,c(2:257)], method = d))
  for (l in 1:k) {
    for (i in 1:y) {
      v<-data.frame(table(t[order(m[x+i,c(1:x)])[c(1:l)],1]))$Freq
      max<-which(v==max(v))
      p$digit[i]<-as.numeric(sample(as.character(max),1))-1
    }
    cm<-table(pt,p$digit)
    er[l]<-100-(sum(diag(cm)) / length(p[,1]) * 100)
  }
  er
}

#call the function and use three different methods 
ert<-predict_knnl(test,train,k=15)
ert_m<-predict_knnl(test,train,d="manhattan",k=15)
ert_c<-predict_knnl(test,train,d="canberra",k=15)

#display error rate 
ert
ert_m
ert_c

#plot results into one graph 
plot(ert,ylim = c(5.4,8.2),xlab = "K",ylab = "Error Rate (in percentage)", main = "Error Rate of Different K under 3 Distance Metric (Test VS Train)",xaxt='n')
axis(1, at = seq(1, 15, by = 1), las=1) 
#https://stackoverflow.com/questions/11775692/how-to-specify-the-actual-x-axis-values-to-plot-as-x-axis-ticks-in-r
points(ert_m,pch = 2)
points(ert_c,pch = 3)
lines(ert)
lines(ert_m,col = "red")
lines(ert_c,col = "green")
legend("topleft",legend = c("Euclidean","Manhattan","Canberra"),col = c("black","red","green"),lty = c(1:3),cex = 0.8,title = "Distance Metric Types")



