% Support Vector Machine function that takes as an input CVSet. As an output it
% gets confusion matrix arr, T2 confusion matrix with labels and t that
% gives in info about time of learning.

function [arr,T2,t] = svmMulti(CVSets)
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
        
        % Learn Support Vector Machine Model giving it values and names and
        % receving SVM Model as a result
        SVMModels = SVMLearn(learningSet.Values, learningSet.Names);
        
        % stop measuring time and save results
        t(i) = toc;

        % Test SVM Model results
        for j = 1:size(testingSet.Values,1)
            % Test Support Vector Machine that takes vector of values from testing
            % set, classification means number of class that model estimates vector belongs
            classification = testSVM(SVMModels, testingSet.Values(j,:), rowName);
            
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

% Learn Support Vector Machine Model giving it values and names and
% receving SVM Model as a result
function SVMModels = SVMLearn(data, names)
    % take classes
    classes = unique(names);
    numClasses = length(classes);
    
    % for every class train SVM model
    for i = 1:numClasses
        indx = strcmp(names, classes(i));
        SVMModels{i} = fitcsvm(data, indx,'ClassNames',[false true],'Standardize',true,...
            'KernelFunction','rbf','KernelScale','auto','BoxConstraint',4,'Solver','SMO');
    end
end

% Test Support Vector Machine that takes vector of values from testing
% set, classification means number of class that model estimates vector belongs
function classification = testSVM(SVMModels, data, classes)
    numClasses = length(classes);
    scores = zeros(1, numClasses);
    
    for i = 1:length(SVMModels)
        [~, score] = predict(SVMModels{i}, data);
        scores(:,i) = score(:,2);
    end
    
    [~, classification] = max(scores,[],2);
end