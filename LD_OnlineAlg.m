% Online learning algorithm using gradient discent for Logistic Discrimination
function[Efnmin,wt_min,Efn] = LD_OnlineAlg(filename,lr,Max_RUN)
[output] = readtable(filename);
features = length(output.Properties.VariableNames)-1;
[input] = output(:,1:features);
[class] = output(:,length(output.Properties.VariableNames));

%converting table to matrix form
x = cell2mat(table2cell(input)); % input
r = cell2mat(table2cell(class)); % output
m = length(r); % size of dataset

% learning rate
   %lr = lr;
   a = -0.01;
   b = 0.01;
    weight = (b-a).*rand(features,1) + a;
    Efn = zeros(Max_RUN, 1); 
    wt = zeros(Max_RUN, 3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for itr = 1:Max_RUN
            effn = 0;
            for p = 1:m
                % linear combination
                lrcom = (x(p,:) * weight);
                lrcom = -1 .* lrcom; 
                % Calculate sigmoid and Efn term
                y = 1/(1+exp(lrcom(1)));
                effn = effn +(r(p,:)*log(y))+((1-r(p,:))*log(1-y)); 
                % gradient calculation
                tx = x'; %transpose(x);
                grad =  tx(:,p) * ((r(p,:) - y)); 

                % weight update
                weight = weight + lr* grad;
                wt(itr,:) = transpose(weight);
            end
            Efn(itr,1) = - effn / 100 ;
        end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Efnmin = Efn(find(Efn==min(Efn)));
    wt_min = wt(find(Efn==min(Efn)),:);
end

