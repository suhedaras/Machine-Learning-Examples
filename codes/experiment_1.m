clc,close all,clear all ;
load fisheriris

%Visualize data
figure
gscatter(meas(:,1),meas(:,2),species,'mbg','o*',8)

%Split the datasets randomly(%80 training, %20 testing)
[m,n] = size(meas) ;
P = 0.80 ;
idx = randperm(m);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_m = meas(idx(1:round(P*m)),:);  % for meas
Testing_m = meas(idx(round(P*m)+1:end),:) ;
Training_s = species(idx(1:round(P*m)),:);  % for species
Testing_s = species(idx(round(P*m)+1:end),:) ;

inds = ~strcmp(Training_s,'setosa'); %removed setosa irises for dual classification
x = Training_m(inds,3:4);
y = Training_s(inds);
inds1 = ~strcmp(Testing_s,'setosa'); %removed setosa irises for dual classification
x_test = Testing_m(inds1,3:4);
y_test = Testing_s(inds1);

SVMModel = fitcsvm(x,y) %trainingleri svm ye soktuk
classOrder = SVMModel.ClassNames
sv = SVMModel.SupportVectors;
figure
gscatter(x(:,1),x(:,2),y,'bg')
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
legend('versicolor','virginica','Support Vector');
hold off

grouphat = predict(SVMModel,x_test ); group = y_test;
[C,order] = confusionmat(group,grouphat,'Order',{'versicolor','virginica'});

TP= C(1,1);FP= C(1,2);FN= C(2,1);TN= C(2,2);

accuracy= (TP+TN)/(TP+TN+FP+FN);
precision=TP/(TP+FP);
recall= TP/(TP+FN);
Fscore = 2*TP*(2*TP + FP + FN); 
TPR= TP/(TP+FN);%(sensitivity)
FPR= TN /(FP + TN);%(specificity)

%Part 3:standardization
for j=1:n
CMean = mean(meas(:,j));
CStd  = std( meas(:,j));    
for i=1:m
meas(i,j)=(meas(i,j)-CMean)/CStd;     
end
end

idx2 = randperm(m);
Training_m2 = meas(idx2(1:round(P*m)),:);  % for meas
Testing_m2 = meas(idx2(round(P*m)+1:end),:) ;
Training_s2 = species(idx2(1:round(P*m)),:);  % for species
Testing_s2 = species(idx2(round(P*m)+1:end),:) ;

inds2 = ~strcmp(Training_s2,'setosa'); %removed setosa irises for dual classification
x2 = Training_m2(inds2,3:4);
y2 = Training_s2(inds2);
inds3 = ~strcmp(Testing_s2,'setosa'); %removed setosa irises for dual classification
x_test2 = Testing_m2(inds3,3:4);
y_test2 = Testing_s2(inds3);

SVMModel2 = fitcsvm(x2,y2) %trainingleri svm ye soktuk
classOrder2 = SVMModel2.ClassNames
sv2 = SVMModel2.SupportVectors;
figure
gscatter(x2(:,1),x2(:,2),y2,'bg')
hold on
plot(sv2(:,1),sv2(:,2),'ko','MarkerSize',10);
title('Scaled fisheriris dataset');
legend('versicolor','virginica','Support Vector');
hold off

grouphat2 = predict(SVMModel2,x_test2 ); group2 = y_test2;
[C2,order2] = confusionmat(group2,grouphat2,'Order',{'versicolor','virginica'})

TP2= C2(1,1);FP2= C2(1,2);FN2= C2(2,1);TN2= C2(2,2);
accuracy2= (TP2+TN2)/(TP2+TN2+FP2+FN2);
precision2=TP2/(TP2+FP2);
recall2= TP2/(TP2+FN2);
Fscore2 = 2*TP2*(2*TP2 + FP2 + FN2); 
TPR2= TP2/(TP2+FN2);%(sensitivity);
FPR2= TN2 /(FP2 + TN2);%(specificity);
%-----------------------------for ionosphere---------------------------
load ionosphere

%Visualize data
figure
gscatter(X(:,19),X(:,20),Y,'bg')

%Split the datasets randomly(%80 training, %20 testing)
[c,d] = size(X) ;
P = 0.80 ;
idx_i = randperm(c);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_mi = X(idx_i(1:round(P*c)),:);  % for meas
Testing_mi = X(idx_i(round(P*c)+1:end),:) ;
Training_si = Y(idx_i(1:round(P*c)),:);  % for species
Testing_si = Y(idx_i(round(P*c)+1:end),:) ;

SVMModel_i = fitcsvm(Training_mi,Training_si) %trainingleri svm ye soktuk
classOrder_i = SVMModel_i.ClassNames
sv_i = SVMModel_i.SupportVectors;
figure
gscatter(Training_mi(:,19),Training_mi(:,20),Training_si,'bg')
hold on
plot(sv_i(:,19),sv_i(:,20),'ko','MarkerSize',10)
legend('b','g','Support Vector')
hold off

grouphat_i = predict(SVMModel_i,Testing_mi ); group_i = Testing_si;
[C_i,order_i] = confusionmat(group_i,grouphat_i,'Order',{'b','g'});

TP_i= C_i(1,1);FP_i= C_i(1,2);FN_i= C_i(2,1);TN_i= C_i(2,2);

accuracy_i= (TP_i+TN_i)/(TP_i+TN_i+FP_i+FN_i);
precision_i=TP_i/(TP_i+FP_i);
recall_i= TP_i/(TP_i+FN_i);
Fscore_i = 2*TP_i*(2*TP_i + FP_i + FN_i); 
TPR_i= TP_i/(TP_i+FN_i);%(sensitivity)
FPR_i= TN_i /(FP_i + TN_i);%(specificity)

%Part 3:standardization
for j=1:d
CMean_i = mean(X(:,j));
CStd_i  = std( X(:,j));    
for i=1:c
 
X(i,j)=(X(i,j)-CMean_i)/CStd_i;     
end
end
for i=1:c
    X(i,2)=[-0.0310];  %for NaN error
end

idx2_i = randperm(c);
Training_m2i = X(idx2_i(1:round(P*c)),:);  % for X
Testing_m2i = X(idx2_i(round(P*c)+1:end),:) ;
Training_s2i = Y(idx2_i(1:round(P*c)),:);  % for Y
Testing_s2i = Y(idx2_i(round(P*c)+1:end),:) ;

SVMModel2_i = fitcsvm(Training_m2i,Training_s2i) %trainingleri svm ye soktuk
classOrder2_i = SVMModel2_i.ClassNames
sv2_i = SVMModel2_i.SupportVectors;
figure
gscatter(Training_m2i(:,19),Training_m2i(:,20),Training_s2i,'bg')
hold on
plot(sv2_i(:,19),sv2_i(:,20),'ko','MarkerSize',10)
title('Scaled ionosphere dataset');
legend('b','g','Support Vector')
hold off

grouphat2_i = predict(SVMModel2_i,Testing_m2i ); group2_i = Testing_s2i;
[C2_i,order2_i] = confusionmat(group2_i,grouphat2_i,'Order',{'b','g'});

TP2_i= C2_i(1,1);FP2_i= C2_i(1,2);FN2_i= C2_i(2,1);TN2_i= C2_i(2,2);
accuracy2_i= (TP2_i+TN2_i)/(TP2_i+TN2_i+FP2_i+FN2_i);
precision2_i=TP2_i/(TP2_i+FP2_i);
recall2_i= TP2_i/(TP2_i+FN2_i);
Fscore2_i = 2*TP2_i*(2*TP2_i + FP2_i + FN2_i); 
TPR2_i= TP2_i/(TP2_i+FN2_i);%(sensitivity)
FPR2_i= TN2_i /(FP2_i + TN2_i);%(specificity)

