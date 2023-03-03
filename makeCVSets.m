%%Creates Cross Validation sets
function CVSets = makeCVSets(values, classes, foldPart)
    uniqueNames = unique(classes);
    nValidations = 1/foldPart;
    nObservations = size(values,1);
    
    for j = 1:nValidations
        %permute observations
        P = randperm(nObservations);
        valuesP = values(P,:);
        classesP = classes(P);
        
        %calculate index of begining of testing set
        testingSize = floor(nObservations*foldPart);
        testingStartIndex = nObservations-testingSize;
        
        %testing set
        tmp = valuesP(testingStartIndex:end,:);
        tmp3 = classesP(testingStartIndex:end);
        
        %learning set
        tmp2 = valuesP(1:testingStartIndex,:);
        tmp4 = classesP(1:testingStartIndex);
        
        %create struct CVSets and save results
        CVSets.TestingSet{j}.Values = tmp;
        CVSets.LearningSet{j}.Values = tmp2;
        CVSets.TestingSet{j}.Names = tmp3;
        CVSets.LearningSet{j}.Names = tmp4;
    end
    CVSets.NValidations = nValidations;
    CVSets.ClassesNames = uniqueNames;
end