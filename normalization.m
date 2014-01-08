function [ trdata_norm, tst_train_data_norm ] = normalization( trdata, tst_train_data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
  tst_train_data_norm=[];
     trdata_norm=[];
            for i= 1: size(trdata,2)
            if (max(trdata(:,i))-min(trdata(:,i)))~= 0
            trdata_norm(:,i)= (trdata(:,i)-min(trdata(:,i)))/(max(trdata(:,i))-min(trdata(:,i)));
            else
                trdata_norm(:,i)= trdata(:,i);
            end
           end

            for i= 1: size(tst_train_data,2)
                if (max(trdata(:,i))-min(trdata(:,i)))~= 0
                tst_train_data_norm(:,i)= (tst_train_data(:,i)-min(trdata(:,i)))/(max(trdata(:,i))-min(trdata(:,i)));
                else
                    tst_train_data_norm(:,i)= tst_train_data(:,i);
                end
            end
            
end

