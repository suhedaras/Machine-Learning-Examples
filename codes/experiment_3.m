clc,close all,clear all ;
load fisheriris
%-------------------------SVM---------------------------------------------------
inds = ~strcmp(species,'setosa'); %removed setosa irises for dual classification
x = meas(inds,3:4);
y = species(inds);
for foldCnt = 5:10
    indices = crossvalind('Kfold',y,foldCnt);
    for i = 1:foldCnt
        test = (indices == i);
        train = ~test;
        SVMModel = fitcsvm(x(train,:),y(train,:));
        sv = SVMModel.SupportVectors;
    end
    figure
    gscatter(x(train,1),x(train,2),y(train,:));
    hold on 
    plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
    hold off
    grouphat = predict(SVMModel,x(test,:) ); group=y(test,:);
    [C_SVM,order] = confusionmat(group,grouphat,'Order',{'versicolor','virginica'});
    TP= C_SVM(1,1);FP= C_SVM(1,2);FN= C_SVM(2,1);TN= C_SVM(2,2);
    precision=TP/(TP+FP)
    recall= TP/(TP+FN)

end
%-------------DCT--------------------------------------
for foldCnt = 5:10
    indices = crossvalind('Kfold',species,foldCnt);
    for i = 1:foldCnt
        test2 = (indices == i);
        train2 = ~test2;
        DCTMdl = fitctree(meas(train2,:),species(train2,:));
    end
      view(DCTMdl,'mode','graph');
      grouphat2 = predict(DCTMdl,meas(test2,:) );  group2=species(test2,:);
      [C_DCT,order2] = confusionmat(group2,grouphat2,'Order',{'setosa','versicolor','virginica'});
end
%-------------kNN methods--------------------------------------
 for foldCnt = 5:10
    indices = crossvalind('Kfold',species,foldCnt);
    for i = 1:foldCnt
        test3 = (indices == i);
        train3 = ~test3;
        KNNMdl=fitcknn(meas(train3,:),species(train3,:),'NumNeighbors',1);   
    end
    grouphat3 = predict(KNNMdl,meas(test3,:)); group3=species(test3,:);
   [C_KNN,order3] = confusionmat(group3,grouphat3,'Order',{'setosa','versicolor','virginica'});
end


%-----------------------------for ionosphere---------------------------
load ionosphere
%-------------------------SVM----------------------
for foldCnt = 5:10
    indices = crossvalind('Kfold',Y,foldCnt);
    for i = 1:foldCnt
        test_i = (indices == i);
        train_i = ~test_i;
        SVMModel_i = fitcsvm(X(train_i,:),Y(train_i,:));
        sv_i = SVMModel_i.SupportVectors;
    end
    figure
    gscatter(X(train_i,19),X(train_i,20),Y(train_i,:));
    hold on 
    plot(sv_i(:,19),sv_i(:,20),'ko','MarkerSize',10);
    hold off
   
    grouphat_i = predict(SVMModel_i,X(test_i,:) ); group_i=Y(test_i,:);
    [Ci_SVM,orderi] = confusionmat(group_i,grouphat_i,'Order',{'b','g'});
    TP_i= Ci_SVM(1,1);FP_i= Ci_SVM(1,2);FN_i= Ci_SVM(2,1);TN_i= Ci_SVM(2,2);
    precision_i=TP_i/(TP_i+FP_i)
    recall_i= TP_i/(TP_i+FN_i)
end
%-------------DCT--------------------------------------
for foldCnt = 5:10
    indices = crossvalind('Kfold',Y,foldCnt);
    for i = 1:foldCnt
        test2_i = (indices == i);
        train2_i = ~test2_i;
        DCTMdl_i = fitctree(X(train2_i,:),Y(train2_i,:));
    end
      view(DCTMdl_i,'mode','graph');
      grouphat2_i = predict(DCTMdl_i,X(test2_i,:) );  group2_i=Y(test2_i,:);
      [Ci_DCT,order2i] = confusionmat(group2_i,grouphat2_i,'Order',{'b','g'});
     
      TP_dct= Ci_DCT(1,1);FP_dct= Ci_DCT(1,2);FN_dct= Ci_DCT(2,1);TN_dct= Ci_DCT(2,2);
      precision_dct=TP_dct/(TP_dct+FP_dct)
      recall_dct= TP_dct/(TP_dct+FN_dct)
end
%-------------kNN methods--------------------------------------
 for foldCnt = 5:10
    indices = crossvalind('Kfold',Y,foldCnt);
    for i = 1:foldCnt
        test3_i = (indices == i);
        train3_i = ~test3_i;
        KNNMdl_i=fitcknn(X(train3_i,:),Y(train3_i,:),'NumNeighbors',1);   
    end
    grouphat3_i = predict(KNNMdl_i,X(test3_i,:)); group3_i=Y(test3_i,:);
   [Ci_KNN,order3i] = confusionmat(group3_i,grouphat3_i,'Order',{'b','g'});
   
    TP_knn= Ci_KNN(1,1);FP_knn= Ci_KNN(1,2);FN_knn= Ci_KNN(2,1);TN_knn= Ci_KNN(2,2);
    precision_knn=TP_knn/(TP_knn+FP_knn)
    recall_knn= TP_knn/(TP_knn+FN_knn)
end




