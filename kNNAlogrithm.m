% kNN Classification that takes CVSets and k parameter, as a result it
% gives confusion matrix arr, T2 as confusion matrix with labels and t -
% time of learning

function [arr, T2,t] = kNNAlogrithm(CVSets, k)
    % init variables
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
        
        % Get kNN model using learning set and k
        knnModel = fitcknn(learningSet.Values, learningSet.Names,'NumNeighbors',k);
        
        % stop measuring time and save results
        t(i) = toc;
        
        % Test kNN model, store results
        for j = 1:size(testingSet.Values,1)
            % Test kNN using trained model, testing set and as a result get classification
            % that means number of class that method estimates vector belongs
            classification = testKnnModel(knnModel, testingSet.Values(j,:), rowName); 
            
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

% Test kNN using trained model, testing set and as a result get classification
% that means number of class that method estimates vector belongs
function classification = testKnnModel(knnModel, data, names)
   y = predict(knnModel,data);
   y = string(y);
   classification = find(names==y);
end