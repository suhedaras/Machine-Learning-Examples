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

resp = strcmp(y,'versicolor'); % resp = 1 or 0 
pred = x;

mdlSVM = fitcsvm(pred,resp,'Standardize',true);
%Compute the posterior probabilities (scores)
mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);
[Xsvm,Ysvm,Tsvm,~,OPTROCPT1] = perfcurve(resp,score_svm(:,mdlSVM.ClassNames),'true');

figure
plot(Xsvm,Ysvm);
hold on
plot(OPTROCPT1(1),OPTROCPT1(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for SVM');
hold off
%---------------------------DCT----------------------------------------
DCTMdl = fitctree(Training_m,Training_s,'ClassNames',{'setosa','versicolor','virginica'});
%--------ROC Curves for DCT
%Predict the class labels and scores for the species
[~,score] = resubPredict(DCTMdl);
%İlk sütun setosa'ya, ikincisi versicolor'a,üçüncüsü virginica'ya karşılık gelir.
%İkili sınıflandırma olmadığı için bu formülden yararlanıcaz--> score(:,2)−max(score(:,1),score(:,3))
diffscore = score(:,2) - max(score(:,1),score(:,3));
[X,Y,T,~,OPTROCPT] = perfcurve(Training_s,diffscore,'versicolor');
% X=FPR, Y=TPR, OPTROCPT=Optimal operating point of the ROC curve
figure
plot(X,Y)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for DCT');
hold off
%---------------------------kNN----------------------------------------
KNNMdl=fitcknn(Training_m,Training_s,'NumNeighbors',4);
%--------ROC Curves for KNN
[~,score3] = resubPredict(KNNMdl);
%İkili sınıflandırma olmadığı için bu formülden yararlanıcaz--> score(:,2)−max(score(:,1),score(:,3))
diffscore3 = score3(:,2) - max(score3(:,1),score3(:,3));
[X3,Y3,T3,~,OPTROCPT3] = perfcurve(Training_s,diffscore3,'versicolor');

figure
plot(X3,Y3)
hold on
plot(OPTROCPT3(1),OPTROCPT3(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for KNN')
hold off

%---------------for ionosphere---------------------------------------------
load ionosphere
%Split the datasets randomly(%80 training, %20 testing)
[c,d] = size(X) ;
P = 0.80 ;
idx_i = randperm(c);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_mi = X(idx_i(1:round(P*c)),:);  % for meas
Testing_mi = X(idx_i(round(P*c)+1:end),:) ;
Training_si = Y(idx_i(1:round(P*c)),:);  % for species
Testing_si = Y(idx_i(round(P*c)+1:end),:) ;

%---------------------------SVM----------------------------------------
resp_i = strcmp(Training_si,'b'); % resp = 1 or 0 
pred_i = Training_mi;

mdlSVM_i = fitcsvm(pred_i,resp_i,'Standardize',true);
%Compute the posterior probabilities (scores)
mdlSVM_i = fitPosterior(mdlSVM_i);
[~,score_svmi] = resubPredict(mdlSVM_i);
[Xsvmi,Ysvmi,Tsvmi,~,OPTROCPT1i] = perfcurve(resp_i,score_svmi(:,mdlSVM_i.ClassNames),'true');

figure
plot(Xsvmi,Ysvmi);
hold on
plot(OPTROCPT1i(1),OPTROCPT1i(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for SVM (ionosphere)');
hold off

%---------------------------DCT----------------------------------------


DCTMdl_i = fitctree(pred_i,resp_i);

% %--------ROC Curves for DCT
% DCTMdl_i = fitPosterior(DCTMdl_i);
[~,score_svmi2] = resubPredict(DCTMdl_i);
[Xsvmi2,Ysvmi2,Tsvmi2,~,OPTROCPT1i2] = perfcurve(resp_i,score_svmi2(:,DCTMdl_i.ClassNames),'true');
figure
plot(Xsvmi2,Ysvmi2);
hold on
plot(OPTROCPT1i2(1),OPTROCPT1i2(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for DCT (ionosphere)');
hold off
% 
% %---------------------------kNN----------------------------------------
KNNMdl_i=fitcknn(pred_i,resp_i,'NumNeighbors',4);
% %--------ROC Curves for KNN
% KNNMdl_i = fitPosterior(KNNMdl_i);
[~,score_svmi3] = resubPredict(KNNMdl_i);
[Xsvmi3,Ysvmi3,Tsvmi3,~,OPTROCPT1i3] = perfcurve(resp_i,score_svmi3(:,KNNMdl_i.ClassNames),'true');
figure
plot(Xsvmi3,Ysvmi3);
hold on
plot(OPTROCPT1i3(1),OPTROCPT1i3(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for KNN (ionosphere)');
hold off


