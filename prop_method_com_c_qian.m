% Developed on 13th May 2011 to compute accuracies with sample weights
% alpha and beta

clear all;
clc;
close all;

addpath('libsvm-mat-weights-2.91-1/libsvm-mat-weights-2.91-1');
% addpath('../netlab');


%% To select dataset
SEMG=0;
newsgroup=0;
sentiment=0;
synthetic=1; 

alpha_weight=1;
laplacian_weight=1;
source_size=0;

compute_alpha =1;

if SEMG==1;
    subjects = {'./matfiles/sub1mw.mat','./matfiles/sub2mw.mat','./matfiles/sub3mw.mat','./matfiles/sub4mw.mat','./matfiles/sub6mw_3.mat','./matfiles/sub7mw_2.mat','./matfiles/sub11mw.mat','./matfiles/sub12mw_3.mat' };  
    %weight = {'./matfiles/alpha_s1_t1.mat','./matfiles/alpha_s2_t1.mat','./matfiles/alpha_s3_t1.mat','./matfiles/alpha_s4_t.mat','./matfiles/sub6mw_3.mat','./matfiles/sub7mw_2.mat','./matfiles/sub11mw.mat','./matfiles/sub12mw_3.mat' };
    
        load (subjects{2});
        O1_X =tr_feat_run;
        O1_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;

        load (subjects{3});  % 6 for our data 8 for Marco
        O2_X =tr_feat_run;
        O2_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;

        load (subjects{4});   %7 for our data 6 for Marco
        O3_X =tr_feat_run;
        O3_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;

        load (subjects{5});  % 5 for our data 7 for Marco
        O4_X =tr_feat_run;
        O4_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;

        load (subjects{6});
        O5_X =tr_feat_run;
        O5_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;

        load (subjects{7});  % 6 for our data 8 for Marco
        O6_X =tr_feat_run;
        O6_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;

        load (subjects{8});   %7 for our data 6 for Marco
        O7_X =tr_feat_run;
        O7_Y = tr_labels;
        clear tr_feat_run;
        clear tr_labels;
        
        %% Test subject

        load(subjects{1});  % test sub 4 is changed to 8 is same
        UT_X= tr_feat_run;
        UT_Y=tr_labels;
        clear tr_feat_run;
        clear tr_labels;
        
        %% To combine the multi source data in one structure
          for i=1:length(subjects)-1;
                        switch(i)
                            case 1
                                S_X{1,i}=O1_X;
                                S_Y{1,i}=O1_Y;
                            case 2
                                S_X{1,i}=O2_X; 
                                S_Y{1,i}=O2_Y;
                            case 3
                                 S_X{1,i}=O3_X;
                                 S_Y{1,i}=O3_Y;
                            case 4
                                 S_X{1,i}=O4_X;
                                 S_Y{1,i}=O4_Y;
                            case 5
                                 S_X{1,i}=O5_X;
                                 S_Y{1,i}=O5_Y;
                            case 6
                                 S_X{1,i}=O6_X;
                                 S_Y{1,i}=O6_Y;
                            case 7
                                 S_X{1,i}=O7_X;
                                 S_Y{1,i}=O7_Y;
                                            
                        end
                        source_size=source_size+size(S_X{1,i},1);
          end
  no_of_models=size(S_X,2);
  no_of_class=length(unique(S_Y{1,1}));        
          
end
  

  
if newsgroup==1
    subjects = {'./newsgrp_data/multidomain/baseball/baseball_10.mat'};
        UT_X=[];
        UT_Y=[];
        for j=1;%length(subjects)
            S=[];
            load (subjects{j});
            UT_X= [UT_X;S.data];
            UT_Y= [UT_Y; S.label];
        end

        load(subjects{1});
        for i=1:size(Td,2)
            S_X{1,i}=Td(1,i).data;
            Sb_Y{1,i}=Td(1,i).label;
            source_size=source_size+size(S_X{1,i},1);
        end
        no_of_models=size(S_X,2);
        no_of_class=length(unique(Sb_Y{1,1}));
         
end


if sentiment ==1
    subjects = {'processed_acl/book_p.mat','processed_acl/dvd_p.mat','processed_acl/electronics_p.mat','processed_acl/kitchen_p.mat'};
    
    % load source data
    load (subjects{4});
   
    O1_X = kit.data;
    O1_Y = kit.label;
    
    load (subjects{2});
    O2_X = dvd.data;
    O2_Y = dvd.label;
    
    load (subjects{1});
    O3_X = book.data;
    O3_Y = book.label;
    
    
    % load target data
    load (subjects{3});
    UT_X = ele.data;
    UT_Y = ele.label;
    
    
    %% To combine the multi source data in one structure
    for i=1:length(subjects)-1;
        switch(i)
            case 1
                S_X{1,i}=O1_X;
                Sb_Y{1,i}=O1_Y;
            case 2
                S_X{1,i}=O2_X; 
                Sb_Y{1,i}=O2_Y;
            case 3
                S_X{1,i}=O3_X;
                Sb_Y{1,i}=O3_Y;
                                      
        end
        source_size=source_size+size(S_X{1,i},1);
    end
    no_of_models=size(S_X,2);
    no_of_class=length(unique(Sb_Y{1,1}));                               
                                                
end

if synthetic ==1
    subjects = {'./matfiles/train1_NIPS_3.mat','./matfiles/train2_NIPS_3.mat','./matfiles/test_NIPS_3.mat'};
    load (subjects{1});    
    O1_X = tr_feat_run;
    O1_Y = tr_labels;
    O1_Y(find(O1_Y==2))=-1;  
    
    tr_feat_run = [];
    tr_labels = [];
    
    load (subjects{2});
    O2_X = tr_feat_run;
    O2_Y = tr_labels;
    O2_Y(find(O2_Y==2))=-1; 
    tr_feat_run = [];
    tr_labels = [];
        
    % load target data
    load (subjects{3});
    UT_X = tr_feat_run;
    UT_Y = tr_labels;
    UT_Y(find(UT_Y==2))=-1;
    
    for i=1:length(subjects)-1;
        switch(i)
            case 1
                S_X{1,i}=O1_X;
                Sb_Y{1,i}=O1_Y;
                
                
            case 2
                S_X{1,i}=O2_X; 
                Sb_Y{1,i}=O2_Y;
                                      
        end
        source_size=source_size+size(S_X{1,i},1);
    end
    no_of_models=size(S_X,2);
    no_of_class=length(unique(Sb_Y{1,1}));     
    
end

% To select classifier used for training and testing
type ='SVM'; 
%type='Ada';  %  %or Ada

sel_class=1; % for binary classification
comb_wt=0;
Diff_weight=1;
weight_u=1;%.01; %0.001;% WEIGHT FOR PSUEDO LABELS
random_weight=0;

 % PARAMETERS For Kernel 
kernel_param= 0.41; %(equiv to gamma= 1/number of features)
Theta=100; %as suggested by paper
percent_l=0.1; % percent of labeled data from target domain
%percent_l_n=0; % 1 for 0.1, 0 for 10%, to control 10 or 1 percent
percent_l_l=[0.1 0.2 0.4 0.6 0.7 0.8 1.0];% 1,2,4,6,7,8,10%
percent_u_u=[0.2 0.4 0.6 0.8 1.0];  % 0,10,20,30,40,50%
mue_array=[0 0.001 0.01 0.1 0.3 0.5 1 100 1000];
percent_u=0.5; % This is kept fixed at 50% percent of unlabeled data from target domain

%% Cross validate the results with 10 different combinations of random data
seeds = [1 2 3 4 5 6 7 8 9 10]; % Seeds for randomizing T_s and S

iter=1;
accuracy_rem_multi=zeros(iter,1);
accuracy_unsuper_multi=zeros(iter,1);
no_of_models=size(S_X,2);
%% To compute alpha weights between each subject and test data
  if alpha_weight==1
      
      if compute_alpha == 1  %% switch how to get alpha  
          doingRealTraining=1;
          regression=0;
          sigma=1000;% 10 for SEMG
          alpha_norm = []; 
          alpha_all=[];
           
            %% Instance Weighting for each subject
         if SEMG==1
             for i= 1: size(UT_X,2)
                 if (max(UT_X(:,i))-min(UT_X(:,i)))~= 0
                    UT_X_norm(:,i)= (UT_X(:,i)-min(UT_X(:,i)))/(max(UT_X(:,i))-min(UT_X(:,i)));
                 else
                    UT_X_norm(:,i)=UT_X(:,i);
                 end
             end
         end
         
         if newsgroup==1
             UT_X_norm=UT_X;
         end
         
         if sentiment==1
             %for i= 1: size(UT_X,2)
             %    if (max(UT_X(:,i))-min(UT_X(:,i)))~= 0
             %       UT_X_norm(:,i)= (UT_X(:,i)-min(UT_X(:,i)))/(max(UT_X(:,i))-min(UT_X(:,i)));
             %    else
             %       UT_X_norm(:,i)=UT_X(:,i);
             %    end
             %end
             UT_X_norm=UT_X;
         end
         
         if synthetic ==1
             UT_X_norm = UT_X;
         end
                   
         for s=1:no_of_models
                predict=[];
                prob=[];
                alpha=[];
                % compute weight for each instance for each source
               % To normalize each trainin

              S_X_norm=[]; 
              if SEMG==1             
                 for i= 1: size(S_X{1,s},2)
                     if (max(S_X{1,s}(:,i))-min(S0_X{1,s}(:,i)))~= 0
                         S_X_norm{1,s}(:,i)= (S_X{1,s}(:,i)-min(S_X{1,s}(:,i)))/(max(S_X{1,s}(:,i))-min(S_X{1,s}(:,i)));
                     else
                         S_X_norm{1,s}(:,i)=S_X{1,s}(:,i);
                     end
                 end           
              end
              
              if newsgroup==1
                S_X_norm{1,s}=S_X{1,s};
              end
              
              if sentiment ==1
                  %for i= 1: size(S_X{1,s},2)
                  %   if (max(S_X{1,s}(:,i))-min(S_X{1,s}(:,i)))~= 0
                  %       S_X_norm{1,s}(:,i)= (S_X{1,s}(:,i)-min(S_X{1,s}(:,i)))/(max(S_X{1,s}(:,i))-min(S_X{1,s}(:,i)));
                  %   else
                  %       S_X_norm{1,s}(:,i)=S_X{1,s}(:,i);
                  %   end
                  %end
                 
                  S_X_norm{1,s}=S_X{1,s};
                  
              end
              if synthetic==1
                  S_X_norm{1,s} = S_X{1,s};
                  
              end
                             
              [alpha EXITFLAG] = MMD(S_X_norm{1,s}, UT_X_norm, sigma, doingRealTraining, regression);
              % Normalize the alpha weights 0 to 1 over each subject
              alpha_s{1,s}=(alpha-repmat(min(alpha),size(alpha,1),1))./(max(alpha)-min(alpha));
              
              % Combine the normalised weights
              alpha_norm=[alpha_norm;alpha_s{1,s}];
         end
         save alpha_ele.mat alpha_s alpha_norm;
          %   save('alpha_baseball4.mat');
         
      else
          load('alpha_ele.mat');
       % load('alpha_basebball1.mat');
      end

  else
      %% set the weights = 1
          alpha_norm=ones(source_size,1);
          for s=1:no_of_models
              alpha_s{1,s}(:,1)=ones(size(S_X{1,s},1),1);
          end
 end


for pert=1:5; %1:length(percent_l_l)
   for pert_u=1;%1:length(percent_u_u)            
            accuracy_rem_cw=[];
           %accuracy_mean_rem=[];   
            accuracy_wt_pro_m= [];   
            accuracy_rem_multi=[];
            pred_unsuper_multi=[];
            accuracy_unsuper_multi=[];
            accuracy_w_mue=[];
            accuracy_rem_class=[];
            %accuracy_rem_mean=[];
        
            for mue_ind=1:length(mue_array) 
                mue=mue_array(mue_ind);
             
                for fold=1:iter

                    L_X=[]; 
                    L_Y=[];
                    U_X=[];
                    U_Y=[];
                    U_X_rand=[];
                    U_Y_rand=[];
                    L_X_t=[];
                    L_Y_t=[];
                    rem_U_X=[];
                    rem_U_Y=[];
                
 

            sel_U_X=[];
            pred_fx=[];
            clust_idx=[];
            A=[];

            clust_idx=[];
            U_X_rand_clust=[];
            Ub_Y_rand_clust=[];
            sel_clust_idx=[];
            L_X_sel_clust_idx=[];
            y_tilde=[];
            y_pred_label_cont=[];
            W_sel_U_X=[];
            prob_estimates_sel=[];
            prob_estimates=[];
            F=[];
            pred_all=[];
            prob_estimates=[];
           
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % select labeled target data
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% select 10% labeled data
            if SEMG==1
               no_of_class=4;
               NC=4;
               L_X_t=[];
               L_Y_t=[];
               [L_X_t,L_Y_t,U_X,U_Y]=select_sample(UT_X,UT_Y,no_of_class,percent_l,seeds(fold));
               L_X=[];
               L_Y=[];
               [L_X,L_Y,U_X_t,U_Y_t]=select_sample(L_X_t,L_Y_t,no_of_class,percent_l_l(pert),seeds(fold));
            end
            if newsgroup==1
               no_of_class=2;
               NC=1;
               L_X_t=[];
               L_Y_t=[];
               [L_X_t,L_Y_t,U_X,U_Y]=select_sample_bin(UT_X,UT_Y,no_of_class,percent_l,seeds(fold));
               L_X=[];
               L_Y=[];
               [L_X,L_Y,U_X_t,U_Y_t]=select_sample_bin(L_X_t,L_Y_t,no_of_class,percent_l_l(pert),seeds(fold));
            end
            if sentiment==1
               no_of_class=2;
               NC=1;
               L_X_t=[];
               L_Y_t=[];
               [L_X_t,L_Y_t,U_X,U_Y]=select_sample_bin(UT_X,UT_Y,no_of_class,percent_l,seeds(fold));
               L_X=[];
  
               [L_X,L_Y,U_X_t,U_Y_t]=select_sample_bin(L_X_t,L_Y_t,no_of_class,percent_l_l(pert),seeds(fold));
            end
            if synthetic ==1
               no_of_class=2;
               NC=1;
               L_X_t=[];
               L_Y_t=[];
               [L_X_t,L_Y_t,U_X,U_Y]=select_sample_bin(UT_X,UT_Y,no_of_class,percent_l,seeds(fold));
               L_X=[];
  
               [L_X,L_Y,U_X_t,U_Y_t]=select_sample_bin(L_X_t,L_Y_t,no_of_class,percent_l_l(pert),seeds(fold)); 
            end

             % randomize tstdata
             [nbData, nbVar] = size(U_X);
             rand_seed = RandStream.create('mrg32k3a','seed',seeds(fold));    
             rand_idx = [];
             a=rand(rand_seed,nbData,1);
             [val ind] = sort(a);
             U_X_rand=U_X(ind,:);
             U_Y_rand= U_Y(ind,:);
             tstsize= round(percent_u*size(U_X,1)); % Test data is kept fixed at 50%
             
             %% From the remaining 50% the unlabeled data is
             %% taken as psuedo labels with varying numbers
             psuedo_size=round(percent_u_u(pert_u)*tstsize);
             sel_U_X=U_X_rand(1:psuedo_size,:);
             sel_U_Y=U_Y_rand(1:psuedo_size,:);
             rem_U_X=U_X_rand(tstsize+1:end,:);
             rem_U_Y=U_Y_rand(tstsize+1:end,:);  
                         
                         if newsgroup==1
                             Lb_Y=L_Y;
                             Ub_Y=U_Y; 
                             rem_Ub_Y=rem_U_Y;
                             sel_Ub_Y=sel_U_Y;
                             Ub_Y_rand=U_Y_rand;
                         end
                         
                         if sentiment==1
                             Lb_Y=L_Y;
                             Ub_Y=U_Y; 
                             rem_Ub_Y=rem_U_Y;
                             sel_Ub_Y=sel_U_Y;
                             Ub_Y_rand=U_Y_rand;
                         end  
                         
                         if synthetic==1
                             Lb_Y=L_Y;
                             Ub_Y=U_Y; 
                             rem_Ub_Y=rem_U_Y;
                             sel_Ub_Y=sel_U_Y;
                             Ub_Y_rand=U_Y_rand;
                         end  
                         
                         

           % To compute accuracy with only labeled data 
                
           [predict,accuracy_L(fold),prob_estimates]= classifier_gen(L_X,L_Y,rem_U_X,rem_U_Y,no_of_class,type);
                      
            %% To compute accuracy when both the weights are zeros            
            
%              if (alpha_weight==0 && laplacian_weight==0)
%                         trdata = [];
%                         trlabel = [];
%              
%                      for i = 1:no_of_models
%                          trdata = [trdata; S_X{1,i}];
%                          trlabel = [trlabel; S_Y{1,i}];               
% 
%                      end
%              
%              %% combine target data into training data
%                          trdata = [trdata; L_X];
%                          trlabel = [trlabel; L_Y]; 
% 
%             if newsgroup==1
%                  model = svmtrain([], trlabel, trdata, '-t 0 -c 10');  % for newsgroup data
%                  [predict, accuracy, prob] = svmpredict(U_Y_rand, U_X_rand, model);
%              end
%              if SEMG==1
%                  [ trdata_new, tstdata_new ] = normalization( trdata, U_X_rand);  
%                  model = svmtrain([], trlabel, trdata_new, '-c 1000 -g 0.5 -e 0.0000001'); % for SEMG data
%                  [predict, accuracy, prob] = svmpredict(U_Y_rand, tstdata_new, model);
%              end       
%                 accuracy_w(fold)=accuracy(1,1); 
%                 
%              end
%    accuracy_rem_mean = mean(accuracy_w);
%          %break; % when weights are zero
              for sel_class=1:NC
                     %%%%%%%for pert_u=1;%1:length(percent_u_u)            %%%%compute binary accuracy of sources on test data%%%%%%%%%%  
                    no_of_class=2; %This number of class is always 2
                  if SEMG==1
                    %% binarise the labels 
                    Sb_Y=[];
                      for i=1:no_of_models
                          Ob_Y=[];                          
                          Ob_Y=S_Y{1,i};
                          Ob_Y(find(S_Y{1,i}== sel_class))=1;
                          Ob_Y(find(S_Y{1,i}~= sel_class))=-1;
                          Sb_Y{1,i}=Ob_Y;
                      end
              
                          UTb_Y=UT_Y;
                          UTb_Y(find(UT_Y == sel_class))=1;
                          UTb_Y(find(UT_Y ~= sel_class))=-1;

                          Ub_Y_rand=U_Y_rand;
                          Ub_Y_rand(find(U_Y_rand == sel_class))=1;
                          Ub_Y_rand(find(U_Y_rand ~= sel_class))=-1;
                                                   
                          rem_Ub_Y=rem_U_Y;
                          rem_Ub_Y(find(rem_U_Y == sel_class))=1;
                          rem_Ub_Y(find(rem_U_Y ~= sel_class))=-1;

                          Lb_Y=L_Y;
                          Lb_Y(find(L_Y==sel_class))=1;
                          Lb_Y(find(L_Y~=sel_class))=-1;

                          sel_Ub_Y=sel_U_Y;
                          sel_Ub_Y(find(sel_U_Y==sel_class))=1;
                          sel_Ub_Y(find(sel_U_Y~=sel_class))=-1;
                  end
          


           %% To compute predicts for each source data on selected
           %% unlabeled tstdata 
              %% Hypothesis Weighting
                W_opt =[];
                if laplacian_weight==1             
               %% predict sel_U_X with weighted SVM
                data=[];
                data=[U_X_rand];
                label=[];
                label=[Ub_Y_rand];%[sel_Ub_Y];
                no_of_models=size(S_X,2);
                prob_estimates=[];
                    for i=1:no_of_models
                        predict=[];
                        prob=[];
                        if newsgroup==1
                           model = svmtrain(alpha_s{1,i},Sb_Y{1,i},S_X{1,i}, '-t 0 -c 10');
                           [predict, accuracy1s, prob] = svmpredict(label, data, model);
                        end
                      if SEMG==1
                           [data, S_X{1,i}] = normalization(data, S_X{1,i});
                           model = svmtrain(alpha_s{1,i},Sb_Y{1,i},S_X{1,i}, '-c 1000 -g 0.5 -e 0.0000001');
                           [predict, accuracy1s, prob] = svmpredict(label, data, model);
                           %[predict,accuracy1s,prob]= classifier_gen_wht(alpha_s{1,i},S_X{1,i},Sb_Y{1,i},data, label, no_of_class,type);
                      end
                      if sentiment==1
                            model = svmtrain(alpha_s{1,i},Sb_Y{1,i},S_X{1,i}, '-t 0 -c 100');
                           [predict, accuracy1s, prob] = svmpredict(label, data, model);
                            
                      end
                      if synthetic ==1
                           model = svmtrain(alpha_s{1,i},Sb_Y{1,i},S_X{1,i}, '-t 0 -c 10 -b 1');
                           [predict, accuracy1s, prob] = svmpredict(label, data, model,'-b 1');
                      end
                        pred_all(:,i)=predict; 
                       % prob_estimates(:,:,i)=prob;
                    end
                    beta=[];
                    [W_opt]= compute_weight_laplacian( U_X_rand, no_of_models, pred_all ); 
                    beta = zeros(size(alpha_norm,1),1);
                    n = 0;
                    for k = 1: no_of_models
                        m = size(S_X{1,k},1);
                        beta(n+1:n+m) = W_opt(k)/m;% comment for newsgroup/m; % sum of beta weight is equal to 1 dividing by size is making it very smallbeta/m;
                        n = n + m;
                    end        
                else
                    beta= ones(size(alpha_norm,1),1);
                end
          
         
              
            
          %% combine weights for instance (alpha) and weights for hypothesis                                    
        
             gamma = alpha_norm .* beta;
             
     %% To compute accuracies with weighted samples        
                         
             %% combine all of the source data
             trdata1 = [];
             trlabel1 = [];
             
%              model_t1 = svmtrain([], Sb_Y{1,1}, S_X{1,1},  '-c 8 -g 0.00078 -e 01');  % without target data unweighted
%              [predict_t1, accuracy_t1, prob] = svmpredict(Ub_Y_rand, U_X_rand, model_t1);
%              
%              model_t2 = svmtrain([], Sb_Y{1,2}, S_X{1,2},  '-c 8 -g 0.00078 -e 01');  % without target data unweighted
%              [predict_t2, accuracy_t2, prob] = svmpredict(Ub_Y_rand, U_X_rand, model_t2);
%               
             
             
             for i = 1:no_of_models
                 trdata1 = [trdata1; S_X{1,i}];
                 trlabel1 = [trlabel1; Sb_Y{1,i}];               
             end
             
%              model_wt1 = svmtrain([], trlabel1, trdata1, '-c 8 -g 0.00078 -e 01');  % without target data unweighted
%              [predict_wt, accuracy_wt1, prob] = svmpredict(Ub_Y_rand, U_X_rand, model_wt1);
%              
%              model_wt2 = svmtrain(gamma, trlabel1, trdata1,'-c 8 -g 0.00078 -e 01');  % without target data weighted
%              [predict_wt, accuracy_wt2, prob] = svmpredict(Ub_Y_rand, U_X_rand, model_wt2);
             %% combine target data into training data
             trdata = [trdata1; L_X];
             trlabel = [trlabel1; Lb_Y];
             
             %% compute weight for test training data
      %       if (laplacian_weight==1)
                gamma = [gamma*mue; ones(size(L_X,1),1)/size(L_X,1)];
%              else
           %     gamma = [gamma; ones(size(L_X,1),1)];
             %end
             
             
                                     
             %% Using weighted SVM for all training data can use S_X_norm
             %% and normalised U_X_rand
          
             
             if newsgroup==1
                model = svmtrain(gamma, trlabel, trdata, '-t 0 -c 10');  % for newsgroup data
                [predict, accuracy, prob] = svmpredict(Ub_Y_rand, U_X_rand, model);
             end
          
             if SEMG==1
                 if (mue==0)
                  [tstdata_new, trdata_new] = normalization( U_X_rand, L_X);    
                  model = svmtrain([],Lb_Y , trdata_new, '-c 1000 -g 0.5 '); % for SEMG data
                 [predict, accuracy, prob] = svmpredict(Ub_Y_rand, tstdata_new, model);    

                 else 
                  [ trdata_new, tstdata_new ] = normalization( trdata, U_X_rand);    
                 model = svmtrain(gamma, trlabel, trdata_new, '-c 1000 -g 0.5 -e 0.0000001'); % for SEMG data
                 [predict, accuracy, prob] = svmpredict(Ub_Y_rand, tstdata_new, model);
                 end
             end   % 
             
             if sentiment==1
                 model = svmtrain(gamma, trlabel, trdata, '-t 0 -c 10');  % for newsgroup data
                 [predict, accuracy, prob] = svmpredict(Ub_Y_rand, U_X_rand, model);
                 
             end
             
              if synthetic==1
                 model = svmtrain(gamma, trlabel, trdata, '-t 0 -c 10');  % for newsgroup data
                 [predict, accuracy, prob] = svmpredict(Ub_Y_rand, U_X_rand, model);
                 
             end
             accuracy_w(fold,sel_class)=accuracy(1,1);
             accuracy_w_mue.data=accuracy_w;           
           %% To compute accuracy as per KDD framework  
             %pred_label=[];
             %pred_label_cont=[];
             %alpha=[];                        
             %[ y_tilde ] = compute_y_tilde_7( W_sel_U_X, no_of_class, sel_class, prob_estimates, Lb_Y);
             %[ alpha1, X1_norm,X1] = compute_alpha( L_X, sel_U_X,y_tilde, Diff_weight, weight_u );
             %[ accuracy_rem, pred_label, pred_label_cont ] = compute_accuracy_belkin7( alpha1, X1_norm, X1, U_X_rand, Ub_Y_rand );
           
             end % end of sel_class                     
       
         end % end of fold
         
         accuracy_rem_class(:,mue_ind) = mean(accuracy_w);
         accuracy_rem_mean(pert,mue_ind) = mean(accuracy_rem_class(:,mue_ind));
         %accuracy_wt_class = mean(accuracy_wt);
         %accuracy_wt_mean = mean(accuracy_wt_class);
         %% To save the results
         %buff=sprintf('sub%d',subno);
         save dvd_w_results.mat accuracy_w accuracy_w_mue accuracy_rem_class accuracy_rem_mean;
         
        end % end of mue

       end % of pert_u
       
    end % end of pert

 
