%%%% ECE 712 
% Final Project
% Sara Jamil
clear; close all; clc;

load('project.mat');

%% OLS - Ordinary Least Squares

Aols = X\Y;

% predicting with Xtst
Ypred = Xtst*Aols;

% test set error
EOLStest = (norm(Ypred-Ytst,'fro')^2)/(norm(Ytst,'fro')^2);
EOLStest_tru = (norm(Ypred-Ytru(251:500,:),'fro')^2)/(norm(Ytru(251:500,:),'fro')^2);

% training set error
Ytrain = X*Aols;
EOLStrain = (norm(Ytrain-Y,'fro')^2)/(norm(Y,'fro')^2);	
EOLStrain_tru = (norm(Ytrain-Ytru(1:250,:),'fro')^2)/(norm(Ytru(1:250,:),'fro')^2);


%% PCA - Principal Component Analysis

for r = 1:10
    Xr = pseudoinv(X,r);
    Apca = Xr*Y;

    % predicting with Xtst
    Ypred = Xtst*Apca;
    
    % test set error
    EPCAtest(r,:) = (norm(Ypred-Ytst,'fro')^2)/(norm(Ytst,'fro')^2);  
    EPCAtest_tru(r,:) = (norm(Ypred-Ytru(251:500,:),'fro')^2)/(norm(Ytru(251:500,:),'fro')^2);
    
    % training set error
    Ytrain = X*Apca;
    
    EPCAtrain(r,:) = (norm(Ytrain-Y,'fro')^2)/(norm(Y,'fro')^2);	
    EPCAtrain_tru(r,:) = (norm(Ytrain-Ytru(1:250,:),'fro')^2)/(norm(Ytru(1:250,:),'fro')^2);
end

figure(1)
plot(EPCAtest);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error');

figure(2)
plot(EPCAtest_tru,'r');
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error_t_r_u');

figure(3)
plot(EPCAtrain,'--');
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error');

figure(4)
plot(EPCAtrain_tru,'r--');
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error_t_r_u');


%% PLS - Partial Least Squares

for r = 1:6
    [U,S,V] = svd(Y'*X,'econ');
    s = diag(S);
    W = V(:,1:r);
    
    T = X*W;
    C = inv(T'*T)*T'*Y;
    
    % predicting with Xtst
    Ttst = Xtst*W;
    Ypred = Ttst*C;
    
    % test set error
    EPLStest(r,:) = (norm(Ypred-Ytst,'fro')^2)/(norm(Ytst,'fro')^2);
    EPLStest_tru(r,:) = (norm(Ypred-Ytru(251:500,:),'fro')^2)/(norm(Ytru(251:500,:),'fro')^2);
        
    % training set error
    Ytrain = T*C;
    
    EPLStrain(r,:) = (norm(Ytrain-Y,'fro')^2)/(norm(Y,'fro')^2);
    EPLStrain_tru(r,:) = (norm(Ytrain-Ytru(1:250,:),'fro')^2)/(norm(Ytru(1:250,:),'fro')^2);
end

figure(5)
plot(EPLStest);
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error');

figure(6)
plot(EPLStest_tru,'r');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error_t_r_u');

figure(7)
plot(EPLStrain,'--');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error');

figure(8)
plot(EPLStrain_tru,'r--');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error_t_r_u');


%% CCA - Canonical Correlation Analysis

for r = 1:6   
%     % This method for finding T and S
%     % provides the same solution as below
%     Sxx = (X'*X)^(-.5);
%     Syy = (Y'*Y)^(-.5);
%     Pxy = Sxx'*X'*Y*Syy;
%     [U,E,V] = svd(Pxy,'econ');
%     U = U(:,1:r);
%     V = V(:,1:r);
%     T = X*Sxx*U;
%     S = Y*Syy*V;
    
    chx = inv(chol(X'*X));
    chy = inv(chol(Y'*Y));
    
    P = chx'*X'*Y*chy;
    
    [U,~,V] = svd(P,'econ');
    U = U(:,1:r);
    V = V(:,1:r);
    
    T = X*chx*U;
    S = Y*chy*V;
    
    A = T\X;
    B = S\Y;
    
    Xr = T*A;
    Yr = S*B;
    
    C = pseudoinv(Xr,r)*Yr;
    
    % predicting with Xtst
    Ypred = Xtst*C;
    
    % test set error
    ECCAtest(r,:) = (norm(Ypred-Ytst,'fro')^2)/(norm(Ytst,'fro')^2);  
    ECCAtest_tru(r,:) = (norm(Ypred-Ytru(251:500,:),'fro')^2)/(norm(Ytru(251:500,:),'fro')^2);
        
    % training set error
    Ytrain = X*C;
    
    ECCAtrain(r,:) = (norm(Ytrain-Y,'fro')^2)/(norm(Y,'fro')^2);	
    ECCAtrain_tru(r,:) = (norm(Ytrain-Ytru(1:250,:),'fro')^2)/(norm(Ytru(1:250,:),'fro')^2);
end

figure(9)
plot(ECCAtest);
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error');

figure(10)
plot(ECCAtest_tru,'r');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error_t_r_u');

figure(11)
plot(ECCAtrain,'--');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error');

figure(12)
plot(ECCAtrain_tru,'r--');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error_t_r_u');


%% Comparing methods

figure(13)
plot(1:6,EPCAtest(1:6),1:6,EPLStest,1:6,ECCAtest);
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error');
legend('PCA','PLS','CCA');

figure(14)
plot(1:6,EPCAtest_tru(1:6),1:6,EPLStest_tru,1:6,ECCAtest_tru);
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Testing Error_t_r_u');
legend('PCA','PLS','CCA');

figure(15)
plot(1:6,EPCAtrain(1:6),'--',1:6,EPLStrain,'--',1:6,ECCAtrain,'--');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error');
legend('PCA','PLS','CCA');

figure(16)
plot(1:6,EPCAtrain_tru(1:6),'--',1:6,EPLStrain_tru,'--',1:6,ECCAtrain_tru,'--');
set(gca,'XTick',1:6);
xlabel('Rank-r'); ylabel('Relative Error');
title('Relative Training Error_t_r_u');
legend('PCA','PLS','CCA');

