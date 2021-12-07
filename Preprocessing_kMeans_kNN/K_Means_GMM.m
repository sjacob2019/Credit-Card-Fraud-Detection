%%
clear;clc;close all;


T = readtable('creditcard.csv');
T = table2array(T);
SMOTE = T(:, 2:end);

%load('SMOTE.mat')


%idx = kmeans(SMOTE(:,1:end-1), k);

% AIC = zeros(1,10);
% GMModels = cell(1,10);
% options = statset('MaxIter',500);
% for k = 1:4
%     GMModels{k} = fitgmdist(SMOTE(:,1:10),k,'Options',options);
%     AIC(k)= GMModels{k}.AIC;
% end
%
% [minAIC,numComponents] = min(AIC);
% numComponents

k = 2;
dVec = 1:29;
for ii = 2:5
    ii
    d = dVec(ii);
    GMModel = fitgmdist(SMOTE(:,[1:d]), k, "CovarianceType","diagonal", "Replicates",20);

    [idx,nlogL,P] = cluster(GMModel,SMOTE(:,[1:d]));

    %idx = kmeans(SMOTE(:,1:end-1), k);

    trueLabel = SMOTE(:, end);

    
    
    maxAccuPlusRcall = 0;
    for i = 1:k-1
        totalIdx = 1:k;
        class1Idx = nchoosek(totalIdx, i);
        for j = 1: length(class1Idx(:, 1))
            class2Idx = totalIdx;
            class2Idx = setdiff(class2Idx, class1Idx(j, :));

            predLabel(ismember(idx, class1Idx(j, :))) = 0;
            predLabel(ismember(idx, class2Idx)) = 1;
            C = confusionmat(trueLabel, predLabel);
            C(1, :) = C(1, :)/sum(C(1, :));
            C(2, :) = C(2, :)/sum(C(2, :));
            [X,Y] = perfcurve(trueLabel,P(:, 2),1);
            if C(1,1) + C(2,2) > maxAccuPlusRcall %&& C(2,2) > 0.85
                maxAccuPlusRcall = C(1,1) + C(2,2);
                C_max = C;
                opt_pred = predLabel';
                [X,Y] = perfcurve(trueLabel,P(:, 1),1);
            end
            
        end
        plot(X, Y)
        hold on
    end
    CVec{ii} = C_max;
    CVec{ii}
    
end
legend('2','3','4','5')

%%
for i = 1:29
    precision(i) = CVec{i}(1,1);
    recall(i) = CVec{i}(2,2);
end

plot(precision)
hold on
plot(recall)


%%
for i = 1:29
    F_score(i) = 2*precision(i)*recall(i) / (precision(i) + recall(i))
end
plot(F_score)
%%
% color = [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980]];
% for i = 0:1
% scatter3(SMOTE((trueLabel == i) & (opt_pred == i), 1), SMOTE(trueLabel == i & opt_pred == i, 2),SMOTE(trueLabel == i & opt_pred == i, 3), '.');
% hold on
% scatter3(SMOTE(trueLabel == 1-i & opt_pred == i, 1), SMOTE(trueLabel == 1-i & opt_pred == i, 2),SMOTE(trueLabel == 1-i & opt_pred == i, 3),'x');
%
% end
%legend('Correctly clustered genuine', 'Incorrectly clustered genuine', 'Correctly clustered fraud', 'Incorrectly clustered fraud')
%grid on;

% scatter3(SMOTE(opt_pred == 0, 1), SMOTE(opt_pred == 0, 2),SMOTE(opt_pred == 0, 3), '.');
% hold on
% scatter3(SMOTE(opt_pred == 1, 1), SMOTE(opt_pred == 1, 2),SMOTE(opt_pred == 1, 3), '.');
%
% legend('Clustered as genuine', 'Clustered as fraud')
%
% set(gca, 'fontsize', 14);
% zlim([-40 10]);
% xlim([-40, 5]);
% ylim([-45, 25]);
% xlabel('1st PCA component');
% ylabel('2nd PCA component');
% zlabel('3nd PCA component');
% view(45 + 90,10)

%%
%%fraudIdx = T{:, 31} == 12genuineIdx = T{:, 31} == 0;
% sum(genuineIdx)
% retainRatio = 1;
% randIdx = randperm(length(genuineIdx));
% genuineIdx(randIdx(1: sum(genuineIdx)*(1 - retainRatio)), :) = false;
% sum(genuineIdx)
%
%
%
%
% %%
% close all;
% sz = 80;
% mkr = 'x';
% curr = figure(1);
% scatter3(T{genuineIdx, 2}, T{genuineIdx, 3}, T{genuineIdx, 4}, sz/100, '.');
% hold on;
% scatter3(T{fraudIdx, 2}, T{fraudIdx, 3}, T{fraudIdx, 4}, sz, mkr);
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
% %%
% function [accuracy, recall] = calcaulateCM(true, est)
% n = length(true);
% accuracy = sum(true == est) / n;
% recall = sum(est(true == 1) == 1) / sum(true == 1);
% end