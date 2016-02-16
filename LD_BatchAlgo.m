% Batch learning algorithm using gradient discent for Logistic Discrimination
function[Jmin,theta_N,J] = LD_BatchAlgo(filename,lr,MAX_ITR)
[output] = readtable(filename);
    features = length(output.Properties.VariableNames)-1;
    [input] = output(:,1:features);
    [class] = output(:,length(output.Properties.VariableNames));

    %converting table to matrix form
    x = cell2mat(table2cell(input));
    y = cell2mat(table2cell(class));
    m = length(y);
    
   a = -0.01;
   b = 0.01;
    theta = (b-a).*rand(features,1) + a;
    J = zeros(MAX_ITR, 1); 
    teta = zeros(MAX_ITR, 3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k = 1:MAX_ITR
        % linear combination and sigmoid function calculation
        h = (1+exp((x * theta).*-1)).^(-1);
        
        % Error calculation
        J(k,1) = -sum((y.*log(h))+((1-y).*log(1-h)))/m; 
        
        %Gradient calculation
        grad = ( x' * (y-h))/m;
        theta = theta + lr .* grad;
        teta(k,:) = theta';
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Jmin = J(find(J==min(J)));
theta_N = teta(find(J==min(J)),:);
end
