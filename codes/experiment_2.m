
clc,close all,clear all ;
load fisheriris
%Split the datasets randomly(%80 training, %20 testing)
[m,n] = size(meas) ;
P = 0.80 ;
idx = randperm(m);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_m = meas(idx(1:round(P*m)),:);  % for meas
Testing_m = meas(idx(round(P*m)+1:end),:) ;
Training_s = species(idx(1:round(P*m)),:);  % for species
Testing_s = species(idx(round(P*m)+1:end),:) ;

%---------------------------SVM----------------------------------------
inds = ~strcmp(Training_s,'setosa'); %removed setosa irises for dual classification
x = Training_m(inds,3:4);
y = Training_s(inds);
inds1 = ~strcmp(Testing_s,'setosa'); %removed setosa irises for dual classification
x_test = Testing_m(inds1,3:4);
y_test = Testing_s(inds1);

SVMModel = fitcsvm(x,y); %trainingleri svm ye soktuk
classOrder = SVMModel.ClassNames;
sv = SVMModel.SupportVectors;
figure
gscatter(x(:,1),x(:,2),y,'bg')
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
title('SVM Model for fisheriris');
legend('versicolor','virginica','Support Vector');
hold off

grouphat = predict(SVMModel,x_test ); group = y_test;
[C_SVM,order] = confusionmat(group,grouphat,'Order',{'versicolor','virginica'})

TP= C_SVM(1,1);FP= C_SVM(1,2);FN= C_SVM(2,1);TN= C_SVM(2,2);

accuracy= (TP+TN)/(TP+TN+FP+FN);
precision=TP/(TP+FP);
recall= TP/(TP+FN);
Fscore = 2*TP*(2*TP + FP + FN); 
TPR= TP/(TP+FN);%(sensitivity)
FPR= FP /(FP + TN);%(specificity)
%---------------------------DCT----------------------------------------
tc = fitctree(Training_m,Training_s);
view(tc,'mode','graph');

grouphat2 = predict(tc,Testing_m );  group2=Testing_s;
[C_DCT,order2] = confusionmat(group2,grouphat2,'Order',{'setosa','versicolor','virginica'})%Konfüzyon matrisi 3x3lüktür,burdanda tp leri bulabiliriz ama svm den farklı bir yöntem ile

[~,score] = resubPredict(tc);
%İlk sütun setosa'ya, ikincisi versicolor'a,üçüncüsü virginica'ya karşılık gelir.
%İkili sınıflandırma olmadığı için bu formülden yararlanıcaz--> score(:,2)−max(score(:,1),score(:,3))
diffscore = score(:,2) - max(score(:,1),score(:,3));
[FPR_dct,TPR_dct,T,AUC_dct] = perfcurve(Training_s,diffscore,'versicolor');
% AUC_dct=accuracy
FPR_dct=mean(FPR_dct);
TPR_dct=mean(TPR_dct);

%---------------------------kNN----------------------------------------
%KNN model oluşturmaz
KNNMdl=fitcknn(Training_m,Training_s,'NumNeighbors',4);

grouphat3 = predict(KNNMdl,Testing_m );  %group2 ile aynı testing_s
[C_KNN,order3] = confusionmat(group2,grouphat3,'Order',{'setosa','versicolor','virginica'})

[~,score3] = resubPredict(KNNMdl);
%İkili sınıflandırma olmadığı için bu formülden yararlanıcaz--> score(:,2)−max(score(:,1),score(:,3))
diffscore3 = score3(:,2) - max(score3(:,1),score3(:,3));
[FPR_knn,TPR_knn,T3,AUC_knn] = perfcurve(Training_s,diffscore3,'versicolor');
% AUC_knn=accuracy
FPR_knn=mean(FPR_knn);
TPR_knn=mean(TPR_knn);
%-----------------------------for ionosphere-------------------------------
load ionosphere
%Split the datasets randomly(%80 training, %20 testing)
[c,d] = size(X) ;
P = 0.80 ;
idx_i = randperm(c);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_mi = X(idx_i(1:round(P*c)),:);  % for meas
Testing_mi = X(idx_i(round(P*c)+1:end),:) ;
Training_si = Y(idx_i(1:round(P*c)),:);  % for species
Testing_si = Y(idx_i(round(P*c)+1:end),:) ;

%---------------------------SVM-----------------------

SVMModel_i = fitcsvm(Training_mi,Training_si); %trainingleri svm ye soktuk
classOrder_i= SVMModel_i.ClassNames;
sv_i = SVMModel_i.SupportVectors;
figure
gscatter(Training_mi(:,19),Training_mi(:,20),Training_si,'bg')
hold on
plot(sv_i(:,19),sv_i(:,20),'ko','MarkerSize',10);
title('SVM Model for ionosphere');
legend('b','g','Support Vector');
hold off

grouphat_i = predict(SVMModel_i,Testing_mi ); group_i = Testing_si;
[Ci_SVM,order_i] = confusionmat(group_i,grouphat_i,'Order',{'b','g'})

TP_i= Ci_SVM(1,1);FP_i= Ci_SVM(1,2);FN_i= Ci_SVM(2,1);TN_i= Ci_SVM(2,2);

accuracy_i= (TP_i+TN_i)/(TP_i+TN_i+FP_i+FN_i);
precision_i=TP_i/(TP_i+FP_i);
recall_i= TP_i/(TP_i+FN_i);
Fscore_i = 2*TP_i*(2*TP_i + FP_i + FN_i); 
TPR_i= TP_i/(TP_i+FN_i);%(sensitivity)
FPR_i= FP_i /(FP_i + TN_i);%(specificity)
%---------------------------DCT----------------------------------------
tc_i = fitctree(Training_mi,Training_si);
view(tc_i,'mode','graph');

grouphat2_i = predict(tc_i,Testing_mi );  group2_i=Testing_si;
[Ci_DCT,order2] = confusionmat(group2_i,grouphat2_i,'Order',{'b','g'})%Konfüzyon matrisi 2x2liktir.

TP_i2= Ci_DCT(1,1);FP_i2= Ci_DCT(1,2);FN_i2= Ci_DCT(2,1);TN_i2= Ci_DCT(2,2);

accuracy_i2= (TP_i2+TN_i2)/(TP_i2+TN_i2+FP_i2+FN_i2);
precision_i2=TP_i2/(TP_i2+FP_i2);
recall_i2= TP_i2/(TP_i2+FN_i2);
Fscore_i2 = 2*TP_i2*(2*TP_i2 + FP_i2 + FN_i2); 
TPR_i2= TP_i2/(TP_i2+FN_i2);%(sensitivity)
FPR_i2= FP_i2 /(FP_i2 + TN_i2);%(specificity)

%---------------------------kNN----------------------------------------
%KNN model oluşturmaz
KNNMdl_i=fitcknn(Training_mi,Training_si,'NumNeighbors',4);

grouphat3_i = predict(KNNMdl_i,Testing_mi );  %group2 ile aynı testing_s
[Ci_KNN,order3] = confusionmat(group2_i,grouphat3_i,'Order',{'b','g'})

TP_i3= Ci_KNN(1,1);FP_i3= Ci_KNN(1,2);FN_i3= Ci_KNN(2,1);TN_i3= Ci_KNN(2,2);

accuracy_i3= (TP_i3+TN_i3)/(TP_i3+TN_i3+FP_i3+FN_i3);
precision_i3=TP_i3/(TP_i3+FP_i3);
recall_i3= TP_i3/(TP_i3+FN_i3);
Fscore_i3 = 2*TP_i3*(2*TP_i3 + FP_i3 + FN_i3); 
TPR_i3= TP_i3/(TP_i3+FN_i3);%(sensitivity)
FPR_i3= FP_i3 /(FP_i3 + TN_i3);%(specificity)

