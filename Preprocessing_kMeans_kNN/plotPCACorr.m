clear;clc;close all;

T = readtable('creditcard.csv');

%%
close all
N = 7;
subIdx = 1;

%corrMatix = corr(T{T{:, end} == 0, 2:N+1});
corrMatix = corr(T{:, 2:N+1});
heatmap(corrMatix)