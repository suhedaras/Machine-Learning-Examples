clc,close all,clear all ;
load fisheriris
%Split the datasets randomly(%80 training, %20 testing)
[m,n] = size(meas) ;
P = 0.80 ;
idx_0 = randperm(m);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_m = meas(idx_0(1:round(P*m)),:);  % for meas
Testing_m = meas(idx_0(round(P*m)+1:end),:) ;
Training_s = species(idx_0(1:round(P*m)),:);  % for species
Testing_s = species(idx_0(round(P*m)+1:end),:) ;

z = Training_m(:,3:4);
%1.Part---------------------------------------------------
clusterNbr=3;
rng(1);
[idx,C]=kmeans(z,clusterNbr);

x1 = min(z(:,1)):0.01:max(z(:,1));
x2 = min(z(:,2)):0.01:max(z(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,3,'MaxIter',1,'Start',C);

figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,[0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(z(:,1),z(:,2),'k*','MarkerSize',5);
plot(C(:,1),C(:,2),'rx','MarkerSize',15,'LineWidth',3);
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Cluster 1','Cluster 2','Cluster 3','Data','Centroids');
hold off;

%2.Part--------------------------------------------------------------------------------

for k=1:6;
rng(1);
[idx_b,Cb]=kmeans(z,k);

x1_b = min(z(:,1)):0.01:max(z(:,1));
x2_b = min(z(:,2)):0.01:max(z(:,2));
[x1G_b,x2G_b] = meshgrid(x1_b,x2_b);
XGrid_b = [x1G_b(:),x2G_b(:)]; % Defines a fine grid on the plot

idx2Region_b = kmeans(XGrid_b,k,'MaxIter',1,'Start',Cb);

figure;
gscatter(XGrid_b(:,1),XGrid_b(:,2),idx2Region_b,[0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(z(:,1),z(:,2),'k*','MarkerSize',5);
plot(Cb(1:k,1),Cb(1:k,2),'rx','MarkerSize',15,'LineWidth',3);
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
hold off;
end
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Data','Centroids');

%----------------------------------------------------------------------------------------------
load ionosphere
%Split the datasets randomly(%80 training, %20 testing)
[c,d] = size(X) ;
P = 0.80 ;
idx_0i = randperm(c);% shuffle the rows, 1 den m ye kadar random satır vektörü oluşturur.(tekrarlayan öğe yok)

Training_mi = X(idx_0i(1:round(P*c)),:);  % for meas
Testing_mi = X(idx_0i(round(P*c)+1:end),:) ;
Training_si = Y(idx_0i(1:round(P*c)),:);  % for species
Testing_si = Y(idx_0i(round(P*c)+1:end),:) ;

zi = Training_mi(:,19:20);

clusterNbri=2;
rng(1);
[idxi,Ci]=kmeans(zi,clusterNbri);

x1i = min(zi(:,1)):0.01:max(zi(:,1));
x2i = min(zi(:,2)):0.01:max(zi(:,2));
[x1Gi,x2Gi] = meshgrid(x1i,x2i);
XGridi = [x1Gi(:),x2Gi(:)]; % Defines a fine grid on the plot

idx2Regioni = kmeans(XGridi,2,'MaxIter',1,'Start',Ci);

figure;
gscatter(XGridi(:,1),XGridi(:,2),idx2Regioni,[0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(zi(:,1),zi(:,2),'k*','MarkerSize',5);
plot(Ci(:,1),Ci(:,2),'rx','MarkerSize',15,'LineWidth',3);
title ('Ionosphere''s Data');
xlabel ('Petal Lengths (cm)');
ylabel ('Petal Widths (cm)'); 
legend('Cluster 1','Cluster 2','Data','Centroids');
hold off;
%2.Part--------------------------------------------------------------------------------

for k=1:6;
rng(1);
[idx_bi,Cbi]=kmeans(zi,k);

x1_bi = min(zi(:,1)):0.01:max(zi(:,1));
x2_bi = min(zi(:,2)):0.01:max(zi(:,2));
[x1G_bi,x2G_bi] = meshgrid(x1_bi,x2_bi);
XGrid_bi = [x1G_bi(:),x2G_bi(:)]; % Defines a fine grid on the plot

idx2Region_bi = kmeans(XGrid_bi,k,'MaxIter',1,'Start',Cbi);

figure;
gscatter(XGrid_bi(:,1),XGrid_bi(:,2),idx2Region_bi,[0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(zi(:,1),zi(:,2),'k*','MarkerSize',5);
plot(Cbi(1:k,1),Cbi(1:k,2),'rx','MarkerSize',15,'LineWidth',3);
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
hold off;
end
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Data','Centroids');



