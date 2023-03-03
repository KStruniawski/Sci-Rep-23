% Random Forest function that takes as an input CVSet and number of tress. 
% As an output it gets confusion matrix arr, T2 confusion matrix with labels and t that
% gives in info about time of learning.


function [arr,T2,t] = randomForest(nTrees, CVSets)
    % starting variables
    nValidations = CVSets.NValidations;
    rowName = CVSets.ClassesNames;
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
        
        % Learn Random Forest giving values, names, number of trees
        RandomForest = TreeBagger(nTrees, learningSet.Values, learningSet.Names, 'Method', ...
            'classification');%',NumPredictorsToSample','all');
        
        % stop measuring time and save results
        t(i) = toc;
        
        % Test Random Forest model, store results
        for j = 1:size(testingSet.Values,1)
            % Test Random Forest that takes vector of values from testing
            % set, classification means number of class that model estimates vector belongs
            classification = RandomForest.predict(testingSet.Values(j,:));
            classification = classification{1};
            classification = find(rowName == classification);
            
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