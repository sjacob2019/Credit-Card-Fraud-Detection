clear;clc;close all;

T = readtable('creditcard.csv');

%%
close all
N = 7;
sz = 1;
subIdx = 1;
for i = (1:N) + 1
    for j = (i+1 : N + 1)
        subplot(N-1, N-1, subIdx);
        subIdx = subIdx + 1;
        scatter(T{T{:, end} == 0, j}, T{T{:, end} == 0, i}, sz,  '.');
        hold on
        scatter(T{T{:, end} == 1, j}, T{T{:, end} == 1, i}, sz, '.');
        if j==i+1
            xlabel("PCA " + string(j-1))
            ylabel("PCA " + string(i-1))
        end
        set(gca, "fontsize", 12)
        set(gca,'xtick',[])
        set(gca,'ytick',[])
    end
    subIdx = subIdx + i - 1;
end