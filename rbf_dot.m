function K = rbf_dot(X, Y,rbf_var)

% Rows of X and Y are data points

xnum = size(X,1);

ynum = size(Y,1);

%if (kernel == 1) % Apply Gaussian kernel
    for i=1:xnum
   %     fprintf('i=%d\n',i);
        for j=1:ynum
            K(i,j) = exp(-norm(X(i,:)-Y(j,:))^2/rbf_var);
           % K(i,j) = X(i,:)*Y(j,:)'; 
        end
    end

% % elseif(kernel==2) % Apply linear kernel
  %   K = X*Y';
% elseif(kernel==2) %polynomial kernel
%     K = (1+X'*Y).^rbf_var;
end