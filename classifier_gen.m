function [ predict_label, accuracy, prob_estimates] = classifier_gen(trdata_new,trlabel_new,tstdata_new,tstlabel_new, no_of_class,type)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

  num=length(tstlabel_new);
  predict_label=zeros(num,1);
  prob_estimates=zeros(num,no_of_class);

  if type =='SVM';
           %%%%%%%%%%%%%% TO Normalize the data       
    %[ trdata_new, tstdata_new ] = normalization( trdata_new, tstdata_new);      
             
     model = svmtrain([],trlabel_new,trdata_new, '-c 1000000 -g 0.5 -e 0.0000001 -b 1');
% 
    [predict_label, accuracy_tr, prob_estimates] = svmpredict(trlabel_new, trdata_new, model,'-b 1');
% 
    [predict_label, accuracy_tst, prob_estimates] = svmpredict(tstlabel_new, tstdata_new, model,'-b 1');
     accuracy = accuracy_tst(1,1);
     
  
% %            % Parameters of SVM.
%             c = 1000000; %inf;   %1000000 ;              %1000;
%             lambda = 1e-7;
%             kerneloption= 2 ; %10 ;%10;%2; %10;   % 2 gives very poor accuracy
%             kernel='gaussian';
%             verbose =0;
%             NC=no_of_class;
% %             
% %        
% %   % Create the SVM classifiers for each of the phases based on features.
% %     
%     xsup=[];
%     ypred_trn=[];
%     
%     [xsup,w,b,nbsv]=svmmulticlassoneagainstall(trdata_new,trlabel_new,NC,c,lambda,kernel,kerneloption,verbose);
%     [ypred_trn] = svmmultival(trdata_new,xsup,w,b,nbsv,kernel,kerneloption);
%     fprintf( '\nRate of correct class in training feature data : %2.2f\n',100*sum(ypred_trn==trlabel_new)/length(trlabel_new)); 
%     [ypred,maxi] = svmmultival(tstdata_new,xsup,w,b,nbsv,kernel,kerneloption);
%     accuracy=100*sum(ypred==tstlabel_new)/length(tstlabel_new);   
%     %pred_class(:,i)=ypred;
%     
%     % To use LIBSVM and get probabilities for each class
%     
% %     model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07 -b 1');
% %     [predict_label, accuracy, prob_estimates] = svmpredict(heart_scale_label, heart_scale_inst, model, '-b 1');
% 
% %%%%Example code%%%%%
% % load heart_scale.mat
% %  model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07 -b 1');
% % [predict_label, accuracy, prob_estimates] = svmpredict(heart_scale_label, heart_scale_inst, model, '-b 1');
% % Accuracy = 86.2963% (233/270) (classification)
% %  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    end 
%          
          
     if type == 'Ada'
      nIter=100;   
         act_model = trainadaboost(trdata_new,trlabel_new,no_of_class,nIter);
         %[error accuracy ypred] =eval_multiclass_boost(tstdata_new,tstlabel_new,act_model,nIter);
          for j=1:length(tstdata_new)
              tstdata_point=tstdata_new(j,:);
             [predict_label(j) prob_estimates(j,:)] = classifyadaboost(tstdata_point,act_model);
           end
          accuracy=100*sum(predict_label==tstlabel_new)/length(tstlabel_new);  
     end 
     







