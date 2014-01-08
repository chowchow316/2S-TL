function [ W_opt ] =compute_weight_laplacian( sel_U_X, no_of_models, A )
%UNTITLED3 Summary of this function goes here
%Detailed explanation goes here
%To form the Laplacian matrix on the target data points (U_X)   
   
  X1=sel_U_X;
  X1_norm=[];
    
  %% normalization
  
%   for i= 1: size(X1,2)  
%       if (max(X1(:,i))-min(X1(:,i)))~= 0
%           X1_norm(:,i)= (X1(:,i)-min(X1(:,i)))/(max(X1(:,i))-min(X1(:,i)));
%       else
%           X1_norm(:,i)= X1(:,i);
%       end
%   end
%   
%      
%     DATA=X1_norm;
    DATA = X1;
    TYPE= 'nn';
    %TYPE = 'epsballs';
    options.NN=13;%6
    options.kernelparam =10; % 0.35
    options.kernel='rbf';
    options.GraphDistanceFunction='euclidean';
    %options.GraphDistanceFunction='epsballs';
    %options.GraphWeights='binary';
    options.GraphWeights='distance';
    options.GraphNormalize=0;
    options.GraphWeightParam=1;
    L_sel = laplacian(DATA, TYPE, options);
    
   %% Let W be the weight matrix for all sources
   %n=length(sel_U_X);
   W_opt=zeros(no_of_models,1);
   %A=[predict1_U_X predict2_U_X];
   H=A'*L_sel*A;
   f=[];
   A1=[];
   b=[];
   Aeq=ones(no_of_models); %[5 5;5 5];
   beq=ones(no_of_models,1); %[5;5];%to say sum X=1
   LB = zeros(no_of_models,1);
   UB =ones(no_of_models,1); %limits 0 and 1
   W_opt = quadprog(H,f,A1,b,Aeq,beq,LB,UB);

end

