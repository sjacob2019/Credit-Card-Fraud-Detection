%% import data
clear;clc;close all;
T = readtable('creditcard.csv');

%% split testing and training
fraudIdx = find(T{:, 31} == 1);
genuineIdx = find(T{:, 31} == 0);

testingRatio = 0.2;

randIdx = randperm(length(fraudIdx));
trainFraudIdx = fraudIdx(randIdx(1: floor(length(fraudIdx)*(1 - testingRatio))));
testFraudIdx = fraudIdx(randIdx(floor(length(fraudIdx)*(1 - testingRatio)) + 1 : end));

randIdx = randperm(length(genuineIdx));
trainGenuineIdx = genuineIdx(randIdx(1: floor(length(genuineIdx)*(1 - testingRatio))));
testGenuineIdx = genuineIdx(randIdx(floor(length(genuineIdx)*(1 - testingRatio)) + 1 : end));

%trainData = T{[trainFraudIdx; trainGenuineIdx], 2:end};
testData = T{[testFraudIdx; testGenuineIdx], 2:end};



%% varying retain ratio/ smote ratio
%% Undersample and SMOTE

retainRatioVec = 10.^(linspace(-4, log10(0.9), 10));
SMOTERatioVec = 10.^(linspace(log10(2),log10(200), 10));

%Vec = SMOTERatioVec;
Vec = retainRatioVec;
for i = 1:length(Vec)

    retainRatio = Vec(i)
    SMOTERatio = 25;

    %SMOTERatio = Vec(i)
    %retainRatio = 0.01;

    randIdx = randperm(length(trainGenuineIdx));
    trainGenuineIdxTemp = trainGenuineIdx;
    trainGenuineIdxTemp(randIdx(1: floor(length(trainGenuineIdx)*(1 - retainRatio))), :) = [];
    trainGenuineData = T{trainGenuineIdxTemp, 2:end};

    trainFraudData = T{trainFraudIdx, 2:end};

    overSampled = smote(trainFraudData(:, 1:end-1), SMOTERatio, ceil(SMOTERatio));

    overSampled = [overSampled, ones(length(overSampled), 1)];
    SMOTE = [overSampled; trainGenuineData];

    %% knn
    kNNModel = fitcknn(SMOTE(:, 1:end-1),SMOTE(:, end), 'NumNeighbors', 5);
    labelPred = predict(kNNModel, testData(:, 1:end-1));
    confusionMatrix = confusionmat(testData(:, end), labelPred);
    confusionMatrixPercent = confusionMatrix ./ [sum(confusionMatrix, 2), sum(confusionMatrix, 2)];
    specificity(i) = confusionMatrix(1,1)/(confusionMatrix(1,1) + confusionMatrix(1,2));
    recall(i) = confusionMatrix(2,2)/(confusionMatrix(2,1) + confusionMatrix(2,2));
end
balancedAccuracy = (specificity + recall)/2;

%%

semilogx((Vec), specificity, 'linewidth', 2)
hold on
semilogx((Vec), recall, 'linewidth', 2)
semilogx((Vec), balancedAccuracy, 'linewidth', 2)
legend('Specificity', 'Recall', 'Balanced Accuracy');
grid on;
set(gca, 'fontsize', 14);
set(gca, 'XDir','reverse')
xlabel('Undersampling Ratio')

% xlabel('SMOTE Ratio')
% xlim([Vec(1), Vec(end)])

%%
% close all;
% sz = 10;
% mkr = 'x';
% curr = figure(1);
% scatter3(T{genuineIdx, 2}, T{genuineIdx, 3}, T{genuineIdx, 4}, sz, '.');
% hold on;
% %scatter3(T{fraudIdx, 2}, T{fraudIdx, 3}, T{fraudIdx, 4}, sz, mkr);
% scatter3(overSampled(:, 1),overSampled(:, 2),overSampled(:, 3), sz, 'x');
% grid on;
% legend('Genuine', 'Fraud');
% set(gca, 'fontsize', 14);
% zlim([-40 10]);
% xlim([-40, 5]);
% ylim([-45, 25]);
% xlabel('1st PCA component');
% ylabel('2nd PCA component');
% zlabel('3nd PCA component');
%
% view(45 + 90,10)
%%
%exportgraphics(curr, ['original.jpg'], 'Resolution',500);
%exportgraphics(curr, ['SMOTE.jpg'], 'Resolution',500);