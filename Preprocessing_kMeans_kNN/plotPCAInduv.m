clear;clc;close all;

T = readtable('creditcard.csv');

%%
close all
N = 7;
subIdx = 1;
for i = (1:N) + 1
    subplot(3,3,i-1)
    [N,edges] = histcounts(T{T{:, end} == 0, i},50, 'Normalization','pdf');
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    plot(edges, N);
    hold on
    [N,edges] = histcounts(T{T{:, end} == 1, i},50, 'Normalization','pdf');
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    plot(edges, N);
end