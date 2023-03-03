% Extreme Learning Machine function that takes as an input CVSet, number of
% neurons and type of activation function on hidden layer. As an output it
% returns confusion matrix arr, T2 confusion matrix with labels and t that
% gives info about time of learning the net.

function [arr,T2,t] = EML(CVSets, nNeurons, activation)
    % starting variables
    rowName = CVSets.ClassesNames;
    nValidations = CVSets.NValidations;
    allResults = cell(nValidations, 1);
    
    % activation function selection
    switch activation
        case 'sigmoid'
            f = @(x)(1 ./ (1 + exp(-x)));
        case 'tanh'
            f = @(x)(tanh(x));
        case 'relu'
            f = @(x)(max(0,x));
        case 'rbf'
            f = @(x)(radbas(x));
        case 'linear'
            f = @(x)(x);
        case 'swish'
            f = @(x)(x ./ (1 + exp(-x)));
        case 'ELiSH'
            f = @(x)(ELiSH(x));
        case 'HardTanH'
            f = @(x)(HardTanH(x));
        case 'TanhRe'
            f = @(x)(TanhRe(x));
        case 'ELUs'
            f = @(x)(ELUs(x));
        case 'Softplus'
            f = @(x)(log(1+exp(x)));
        case 'LReLU'
            f = @(x)(LReLU(x));
        case 'BinaryStep'
            f = @(x)(BinaryStep(x));
    end

    % loop throught the all CVSets
    for i = 1:nValidations
        % init results vector
        results = zeros(length(rowName),  length(rowName));
        
        % get learning and testing set
        learningSet = CVSets.LearningSet{i};
        testingSet = CVSets.TestingSet{i};

        % start measuring time
        tic
        
        % learn ELM network giving values, names, number of neurons,
        % activation function f and as a result we get wages w between 
        % input and hidden layer (random), bias(random) and beta wages
        % that are calculated using pseudo-inverse operation
        [w,b,beta] = EMLNetLearn(learningSet.Values, learningSet.Names, nNeurons, f);
        
        % stop measuring time and save results
        t(i) = toc;
        
        %test network iterating throught the testing set, store results
        for j = 1:size(testingSet.Values,1)
            % test EML network that takes vector of values from testing
            % set, learnt EML parameters (wages, bias, beta, number of
            % neurons and activation function) and gives classification
            % that means number of class that net estimates vector belongs
            classification = EMLNetTest(testingSet.Values(j,:)', w, b, beta, nNeurons, f);
            
            % calculations to build confusion matrix
            row = find(rowName == testingSet.Names(j));
            results(row, classification) = results(row, classification) + 1;
        end
        
        % build confusion matrix with labels
        rowNameI = strcat(rowName, "_" + i);
        allResults{i} = array2table(results,'RowNames',rowNameI,'VariableNames',rowName);
    end
    
    % sum up all results from the loop to the one table
    arr = makeASummary(allResults);
    T2 = array2table(arr,'RowNames',rowName,'VariableNames',rowName);
    t = mean(t);
end

% learn EML network giving values, names, number of neurons,
% activation function f and as a result we get wages, bias and beta
% that are EML parameters
function [w, b, beta] = EMLNetLearn(data, names, nNeurons, f)
    data = data'; 
    N = size(data,2); % number of training samples
    d = size(data,1); % length of sample
    L = nNeurons;   % number of hidden units
        
    % create output layer
    uNames = unique(names); % get classes names
    [~,~,idxMap] = unique(names); % create map of classes
    T = zeros(size(data,2),length(uNames)); % init output layer
    for i = 1:size(data,2)
        T(i,idxMap(i)) = 1; % fill output layer with classification true results
    end
    
    % create random wages between input and hidden layer and random bias
    w = unifrnd(-1,1,[d,L]);
    b = unifrnd(-1,1,[N,1]);
    H = zeros(N,L);
    
    % make calculations in hidden layer using value of activation function f
    % of dot product of input vector and random wages increased by bias, 
    % results are output values from hidden layer
    for i=1:N
        for j=1:L
            H(i,j) = f(dot(w(:,j),data(:,i)) + b(i));
        end
    end
    
    % make pseudoinverse, the result are wages between hidden and output
    % layer
    beta = pinv(H)*T;
end

% test EML network that takes vector of values from testing
% set, learnt EML parameters (wages, bias, beta, number of
% neurons and activation function) and gives classification
% that means number of class that net estimates vector belongs
function classification = EMLNetTest(data, w, b, beta, nNeurons, f)
    N = size(data,2); % number of training samples
    d = size(data,1); % length of sample
    L = nNeurons;%number of hidden units
    
    % make calculations in hidden layer using value of activation function f
    % of dot product of input vector and wages increased by bias, 
    % results are output values from hidden layer
    for i=1:N
        for j=1:L
            H(i,j) = f(dot(w(:,j),data(:,i)) + b(i));
        end
    end
    
    %output layer
    z = H*beta;
    z = f(z);
    
    % output layer results
    [~,classification] = max(z);
end

%% activation functions
function y = ELiSH(x)
    for i = 1:length(x)
        if x(i) >= 0
            y(i) = x(i) / (1+exp(-x(i)));
        else
            y(i) = (exp(x(i))-1) / (1+exp(-x(i)));
        end
    end
end

function y = HardTanH(x)
    for i = 1:length(x)
        if x(i) < -1
            y(i) = -1;
        elseif x(i) >= -1 && x(i) <= 1
            y(i) = x(i);
        else
            y(i) = 1;
        end
    end
end

function y = TanhRe(x)
    for i = 1:length(x)
        if x(i) > 0
            y(i) = x(i);
        else
            y(i) = tanh(x(i));
        end
    end
end

function y = ELUs(x)
    for i = 1:length(x)
        if x(i) > 0
            y(i) = x(i);
        else
            y(i) = exp(x(i))-1;
        end
    end
end

function y = LReLU(x)
    for i = 1:length(x)
        if x(i) > 0
            y(i) = x(i);
        else
            y(i) = 0.01*x(i);
        end
    end
end

function y = BinaryStep(x)
    for i = 1:length(x)
        if x(i) >= 0
            y(i) = 1;
        else
            y(i) = 0;
        end
    end
end