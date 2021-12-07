%%
clear;clc;close all;
T = readtable('creditcard.csv');

%%
fraudIdx = T{:, 31} == 1;
genuineIdx = T{:, 31} == 0;
sum(genuineIdx)
retainRatio = 0.1;
randIdx = randperm(length(genuineIdx));
genuineIdx(randIdx(1: sum(genuineIdx)*(1 - retainRatio)), :) = false;
sum(genuineIdx)

%%
fraudData = T{fraudIdx, 2:30};

overSampled = smote(fraudData, 10, 10);
genuineData = T{genuineIdx, 2:31};
overSampled = [overSampled, ones(length(overSampled), 1)];
SMOTE = [overSampled; genuineData];
%%
close all;
sz = 10;
mkr = 'x';
curr = figure(1);
scatter3(T{genuineIdx, 2}, T{genuineIdx, 3}, T{genuineIdx, 4}, sz, '.');
hold on;
%scatter3(T{fraudIdx, 2}, T{fraudIdx, 3}, T{fraudIdx, 4}, sz, mkr);
scatter3(overSampled(:, 1),overSampled(:, 2),overSampled(:, 3), sz, 'x');
grid on;
legend('Genuine', 'Fraud');
set(gca, 'fontsize', 14);
zlim([-40 10]);
xlim([-40, 5]);
ylim([-45, 25]);
xlabel('1st PCA component');
ylabel('2nd PCA component');
zlabel('3nd PCA component');

view(45 + 90,10)
%%
%exportgraphics(curr, ['original.jpg'], 'Resolution',500);
%exportgraphics(curr, ['SMOTE.jpg'], 'Resolution',500);