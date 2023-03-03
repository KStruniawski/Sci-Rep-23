% makeClassification - input dataFile.mat with calculated features by 
% makeFeaturesFromImagesAndMasks and for given feature selection method

function results = makeClassification(featuresFileName, selectionMethod)
%     cd("C:\Users\Karol\Desktop\fspackage");
%     load_fspackage;
%     javaaddpath('C:\Program Files\Polyspace\R2021a\java\jar');

    %% load from file assuming T is path or based on passed features values given by T
    path = "data/mat/" + featuresFileName + ".mat";
    T = load(path).T;
    
    data = table2cell(T);
    data(:,1) = [];
    data = cell2mat(data);
    
    %% make features selection (get indexes of features to be removed)
    classes = string(T.class);
    [~, ~, classesIndexes] = unique(classes);
     
    switch selectionMethod
        case 'fcbf'
            s = fsFCBF(data, classesIndexes);
        case 'sbmlr'
            s = fsSBMLR(data, classesIndexes);
        case 'cfs'
            s = fsCFS(data, classesIndexes);
        case 'infogain'
            s = fsInfoGain(data, classesIndexes);
        case 'mrmr'
            [idx, scores] = fscmrmr(data, classesIndexes);
            for i = 1:length(idx)
                if scores(idx(i)) > 0.01
                    s.fList(i) = idx(i);
                else
                    break
                end
            end
        otherwise
            s.fList = 1:size(data,2);
    end

    indexesToStay = s.fList;
    indexesToStay2 = indexesToStay + 1;
    indexesToRemove = 1:size(data,2);
    indexesToRemove(indexesToStay) = [];
    indexesToRemove2 = indexesToRemove + 1;
        
    %print what is removed
    fprintf('\nFeatures to be removed:');
    for i = 1:length(indexesToRemove2)
        fprintf('\n\t%i: %s', i, string(T.Properties.VariableNames(indexesToRemove2(i))));
    end
    removedFeatures = string(T.Properties.VariableNames(indexesToRemove2));
    presentFeatures = string(T.Properties.VariableNames(indexesToStay2));
    save("data/mat/RemovedFeatures.mat", "removedFeatures");
    save("data/mat/RemainedFeatures.mat", "presentFeatures");
    
    %remove features
    dataAfterSelection = data;
    dataAfterSelection(:,indexesToRemove) = [];
    TAfterSelection = T;
    TAfterSelection(:,indexesToRemove2) = [];
    save("data/mat/FeaturesTableAfterSelection.mat", "TAfterSelection");
    save("data/mat/FeaturesValueAfterSelection.mat", "dataAfterSelection");
    classifications = string(TAfterSelection.class);
    save("data/mat/DataClassifications.mat", "classifications");
    
    %print features present in model
    fprintf('\nFeatures after selection:');
    for i = 1:length(TAfterSelection.Properties.VariableNames)
        fprintf('\n\t%i: %s', i, string(TAfterSelection.Properties.VariableNames(i)));
    end
    
    %% ---Machine Learning parameters---%%
    validPart = 0.1;    %between 0 and 1
    nTrees = 200;   % RF
    mlpTopology = [33,33,33];   % MLP
%     rbnTopology = [11,12,11];   % RBN
%     rbnCentroidsNum = 25;    % RBN
    emlNumNeurons = 3900;   % EML
%     emlRbnNumNeurons = 1250;    % EML-RBN 
%     emlRbnNumCentroids = 36;     % EML-RBN 
    numIterations = 100;%10;
    %%---parameters to change---%%
    
    RELMC = 2.8;
    RELMNeurons = 3900;
    results = [];

    fprintf('\n\n\n---------Valid part is %f---------\n\n\n\n', validPart);
   
    for c = 1:length(RELMC)
        for n = 1:length(RELMNeurons)
            parfor i = 1:numIterations
            %% make CrossValidation sets
            CVSets = makeCVSets(dataAfterSelection, classifications, validPart);
            %For augmented data
            %CVSets = makeCVSetsAugment(dataAfterSelection, classifications, fileNames, validPart);

            %% make SVM MultiClass Model Classification
    %         [confMatrixSVM_arr, confMatrixSVM_T,tSVM(i)] = svmMulti(CVSets);
    %         [ResultSVM, ~]=confusion.getValues(confMatrixSVM_arr);
    % %         disp(ResultSVM)
    %         SVMResultsAccuracy(i) = ResultSVM.Accuracy;
    %         ConfMatrixesSVM{i} = confMatrixSVM_T;
    %         SVMResults{i} = ResultSVM;

    %         %% make Random Forest Classification
    %         [confMatrixRF_arr, confMatrixRF_T,tRF(i)] = randomForest(nTrees,CVSets);
    %         [ResultRF, ~] = confusion.getValues(confMatrixRF_arr);
    %         RFResultsAccuracy(i) = ResultRF.Accuracy;
    %         ConfMatrixesRF{i} = confMatrixRF_T;
    %         RFResults{i} = ResultRF;
    % % 
    %         %% make k-NN Classification (with selection of the best k between 1 and 5)
    %         [confMatrixKnn,confMatrixKnn_T,tKNN(i)] = kNNAlogrithm(CVSets, 1);
    %         [ResultKnn, ~]=confusion.getValues(confMatrixKnn);
    %         KNNResultsAccuracy(i) = ResultKnn.Accuracy;
    %         ConfMatrixesKNN{i} = confMatrixKnn_T;
    %         KNNResults{i} = ResultKnn;
    % 
    %         %% make MLP Net for Classification
    %         [confMatrixMLP_arr, confMatrixMLP_T,tMLP(i)] = MLP(CVSets, mlpTopology);
    %         [ResultMLP, ~]=confusion.getValues(confMatrixMLP_arr);
    % %         disp(ResultMLP)
    %         MLPResultsAccuracy(i) = ResultMLP.Accuracy;
    %         ConfMatrixesMLP{i} = confMatrixMLP_T;
    %         MLPResults{i} = ResultMLP;
    %         
    %         %% make Radial Basis Net for Classification
    %         [confMatrixRBN_arr, confMatrixRBN_T,tRBN(i)] = RBN(CVSets, rbnCentroidsNum, rbnTopology);
    %         [ResultRBN, ~]=confusion.getValues(confMatrixRBN_arr);
    % %         disp(ResultRBN)
    %         RBNResultsAccuracy(i) = ResultRBN.Accuracy;
    %         ConfMatrixesRBN{i} = confMatrixRBN_T;
    %         RBNResults{i} = ResultRBN;

%             %% make Extreme Learning Machine for Classification
%             [confMatrixEML_arr, confMatrixEML_T,tEML(i)] = EML(CVSets, emlNumNeurons, 'LReLU');
%             [ResultEML, ~]=confusion.getValues(confMatrixEML_arr);
%     %         disp(ResultEML)
%             EMLResultsAccuracy(i) = ResultEML.Accuracy;
%             ConfMatrixesEML{i} = confMatrixEML_T;
%             EMLResults{i} = ResultEML;
    %         
    %         %% make Radial Basis Function - Extreme Learning Machine for Classification
    %         [confMatrixRBN_EML_arr, confMatrixRBN_EML_T, tRBN_EML(i)] = RBN_EML(CVSets,emlRbnNumNeurons,...
    %             emlRbnNumCentroids,'linear');
    %         [ResultRBN_EML, ~]=confusion.getValues(confMatrixRBN_EML_arr);
    % %         disp(ResultRBN_EML)
    %         RBN_EMLResultsAccuracy(i) = ResultRBN_EML.Accuracy;
    %         ConfMatrixesRBN_EML{i} = confMatrixRBN_EML_T;
    %         RBN_EMLResults{i} = ResultRBN_EML;
% 
            %% make Regularized Extreme Learning Machine for Classification
            [confMatrixREML_arr, confMatrixREML_T,tREML(i)] = RELM(CVSets, RELMNeurons(n), 'LReLU', RELMC(c));
            [ResultREML, ~]=confusion.getValues(confMatrixREML_arr);
            REMLResultsAccuracy(i) = ResultREML.Accuracy;
            ConfMatrixesREML{i} = confMatrixREML_T;
            REMLResults{i} = ResultREML;
            end
%             res(n) = mean(REMLResultsAccuracy);
        end
%         results = [results;res];
    end
end