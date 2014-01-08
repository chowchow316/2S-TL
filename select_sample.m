function [L_X,L_Y,U_X,U_Y]=select_sample(tstdata,tstlabel,no_of_class,tstpercent,fold)

% This program selects percent of data from data having no_of_class
% randomly such that equal number of data get selected from each class.
seeds = [1 2 3 4 5 6 7 8 9 10]; % Seeds for randomizing T_s and S
% randomize tstdata
        [nbData, nbVar] = size(tstdata);
        rand_seed = RandStream.create('mrg32k3a','seed',seeds(fold));    
        rand_idx = [];
        a=rand(rand_seed,nbData,1);
        [val ind] = sort(a);
        tstdata_rand=tstdata(ind,:);
        tstlabel_rand=tstlabel(ind,:);

 % % To initialise the training data with 10% of training data from tstdata
       
        %tstpercent=0.1;
        tstsize= round(tstpercent*size(tstdata,1));
        
        % to take equal no of data belonging to each class from the test
        % data
        tstsize_class= round(tstsize/no_of_class);          
        
        % to separate the tst data belonging to separate classes
           %%%%%%%%%to find the phase data for test data%%%%
        
        [X1T Y1]= find(tstlabel_rand==1);
        ind1=ind(X1T);
        [X2T Y2]= find(tstlabel_rand==2);
        ind2=ind(X2T);
        [X3T Y3]= find(tstlabel_rand==3);
        ind3=ind(X3T);
        [X4T Y4]= find(tstlabel_rand==4);
        ind4=ind(X4T);
        
         % to check if data from all labels are there in tst_train_data
          
          if length(X1T)==0 
               disp(sprintf('label1 not present in  tst data'));
          end
          if length(X2T)==0
               disp(sprintf('label2 not present in  tst data'));
          end 
          if length(X3T)==0
                disp(sprintf('label3 not present in  tst data'));
          end
          if length(X4T)==0
                disp(sprintf('label4 not present in  tst data'));
          end
                               
                 
         tstdata_1=tstdata_rand(X1T,:);
         tstdata_2=tstdata_rand(X2T,:);
         tstdata_3=tstdata_rand(X3T,:) ;         
         tstdata_4=tstdata_rand(X4T,:) ;
                   
         if length(tstdata_1) < tstsize_class
             tst_train_data_1=tstdata_1;
             tst_train_idx_1=ind(X1T,1);
             tstdata_new1=[];
         else
             tst_train_data_1=tstdata_1(1:tstsize_class,:);
             tst_train_idx_1=X1T(1:tstsize_class,:);
             tstdata_new1=tstdata_1(tstsize_class+1:end,:);
         end
         if length(tstdata_2) < tstsize_class
             tst_train_data_2=tstdata_2;
              tstdata_new2=[];
         else
             tst_train_data_2=tstdata_2(1:tstsize_class,:);
             tstdata_new2=tstdata_2(tstsize_class+1:end,:);
         end
         
         if length(tstdata_3) < tstsize_class
             tst_train_data_3=tstdata_3;
              tstdata_new3=[];
         else
             tst_train_data_3=tstdata_3(1:tstsize_class,:);
             tstdata_new3=tstdata_3(tstsize_class+1:end,:);
         end
         
         if length(tstdata_4) < tstsize_class
             tst_train_data_4=tstdata_4;
              tstdata_new4=[];
         else
             tst_train_data_4=tstdata_4(1:tstsize_class,:);
             tstdata_new4=tstdata_4(tstsize_class+1:end,:);
         end
                 
      tst_train_data=[tst_train_data_1;tst_train_data_2;tst_train_data_3;tst_train_data_4];
      tst_train_label=[ones(size(tst_train_data_1,1),1); ones(size(tst_train_data_2,1),1)*2;ones(size(tst_train_data_3,1),1)*3;ones(size(tst_train_data_4,1),1)*4];
    
      tstdata_new=[tstdata_new1; tstdata_new2;tstdata_new3;tstdata_new4];
      tstlabel_new=[ones(size(tstdata_new1,1),1); ones(size(tstdata_new2,1),1)*2;ones(size(tstdata_new3,1),1)*3;ones(size(tstdata_new4,1),1)*4];
     
     
      
      %%%%%%%%%%%%%%%%%%%%%To initialise the variables%%%%%%%%%
     
     L_X=tst_train_data;
     L_Y=tst_train_label;
     U_X= tstdata_new;
     U_Y=tstlabel_new;
     
