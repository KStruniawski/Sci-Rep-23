% Multilayer Perceptron function that takes as an input CVSet and topology of the net. 
%As an output it gets confusion matrix arr, T2 confusion matrix with labels and t that
% gives in info about time of learning the net.


function [arr,T2,t] = MLP(CVSets, topology)
    % starting variables
    rowName = CVSets.ClassesNames;
    nValidations = CVSets.NValidations;
    allResults = cell(nValidations, 1);

    % loop throught the all CVSets
    for i = 1:nValidations
        % init results vector
        results = zeros(length(rowName),  length(rowName));
        
        % get learning and testing set
        learningSet = CVSets.LearningSet{i};
        testingSet = CVSets.TestingSet{i};
   
        % start measuring time
        tic
        
        % learn MLP network giving values, names, topology
        MLPNet = MLPNetLearn(learningSet.Values, learningSet.Names, topology);
        
        % stop measuring time and save results
        t(i) = toc;
        
        % Test network, store results
        for j = 1:size(testingSet.Values,1)
            % test MLP network that takes vector of values from testing
            % set, classification means number of class that net estimates vector belongs
            classification = vec2ind(MLPNet(testingSet.Values(j,:)'));
            
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

% Learn MLP network giving values, names, topology
function MLPNet = MLPNetLearn(data, names, topology)
    data = data'; 
    uNames = unique(names);
    MLPNet = feedforwardnet(topology);
    MLPNet.trainParam.showWindow = 0; 
    [~,~,idxMap] = unique(names);
    target = zeros(size(data,2),length(uNames));
    for i = 1:size(data,2)
        target(i,idxMap(i)) = 1;
    end
    MLPNet = train(MLPNet,data,target');
end