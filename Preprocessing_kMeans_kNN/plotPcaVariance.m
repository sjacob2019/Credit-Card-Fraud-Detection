clear;clc;close all;

T = readtable('creditcard.csv');

cumtemp = 0;
for i = 2:29
    varianceOri(i-1) = var(T{:, i});
    cumtemp = cumtemp + varianceOri(i-1);
    cumVarOri(i-1) = cumtemp;
end

cumtemp = 0;
load('SMOTE.mat')
for i = 1:28
    varianceSMOTE(i) = var(SMOTE(:, i));
    cumtemp = cumtemp + varianceSMOTE(i);
    cumVarSMOTE(i) = cumtemp;
end

totalVarOri = sum(varianceOri);
totalVarSMOTE = sum(varianceSMOTE);


%%
close all
x = 1:28;
plot(x, cumVarOri/totalVarOri*100, "lineWidth", 2)
hold on
plot(x, cumVarSMOTE/totalVarSMOTE*100,  "lineWidth", 2)
set(gca, "fontsize", 16)
xlim([1, 28]);
grid on
xticks(1:28)
%xticklabels(repmat("PCA ",1, 28) + string(1:28))
xlabel("PCA Component")
yticks(0:20:100)
ylabel("Cumulative total variance (%)")

legend("Original", "SMOTE")