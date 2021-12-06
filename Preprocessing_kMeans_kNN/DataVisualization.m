%%
clear;clc;close all;
T = readtable('creditcard.csv');

%%
fraudIdx = T{:, 31} == 1;
genuineIdx = T{:, 31} == 0;
sum(genuineIdx)
retainRatio = 1;
randIdx = randperm(length(genuineIdx));
genuineIdx(randIdx(1: sum(genuineIdx)*(1 - retainRatio)), :) = false;
sum(genuineIdx)
%%
close all;
sz = 80;
mkr = 'x';
curr = figure(1);
scatter3(T{genuineIdx, 2}, T{genuineIdx, 3}, T{genuineIdx, 4}, sz/100, '.');
hold on;
scatter3(T{fraudIdx, 2}, T{fraudIdx, 3}, T{fraudIdx, 4}, sz, mkr);
grid on;
legend('Genuine', 'Fraud');
set(gca, 'fontsize', 14);
zlim([-40 10]);
xlim([-40, 5]);
ylim([-45, 25]);
xlabel('1st PCA component');
ylabel('2nd PCA component');
zlabel('3nd PCA component');

%%
exportgraphics(curr, ['PCA', num2str(retainRatio), '.jpg'], 'Resolution',300);