clear;clc;close all;

T = readtable('creditcard.csv');
numOfFraud = sum(T{:, end} == 1);
numOfGenuine = sum(T{:, end} == 0);

underSampingRatio = 1;

GenuineData = T{T{:, end} == 0, :};
underSampledGenuineData = GenuineData(randsample(numOfGenuine, numOfFraud * underSampingRatio), :);

FraudData = T{T{:, end} == 1, :};

underSampledData = [underSampledGenuineData;FraudData];