%% Generate PR Curves
% Authors: Tzofi Klinghoffer, Caleb Perez
% Description: Generates precision-recall curves by class for two different
% datasets (e.g. a full data set and a cropped data set). Reads data from
% two Excel files, formatted as described below.

clear all; close all; clc;

%% Read Data
% Coded for two spreadsheets, cropped and full, with the first two columns
% recall and precision for scallops, the next two for roundfish, then
% flatfish, then skates (8 columns total per Excel file).

data1 = xlsread('CroppedPR.xlsx');  % Replace with Excel file name
Scal1 = data1(:,1:2);
Scal1 = Scal1(~any(isnan(Scal1),2),:);
Round1 = data1(:,3:4);
Round1 = Round1(~any(isnan(Round1), 2),:);
Flat1 = data1(:,5:6);
Flat1 = Flat1(~any(isnan(Flat1), 2),:);
Skate1 = data1(:,7:8);
Skate1 = Skate1(~any(isnan(Skate1), 2),:);

data2 = xlsread('FullPR.xlsx'); % Replace with Excel file name
Scal2 = data2(:,1:2);
Scal2 = Scal2(~any(isnan(Scal2), 2),:);
Round2 = data2(:,3:4);
Round2 = Round2(~any(isnan(Round2), 2),:);
Flat2 = data2(:,5:6);
Flat2 = Flat2(~any(isnan(Flat2), 2),:);
Skate2 = data2(:,7:8);
Skate2 = Skate2(~any(isnan(Skate2), 2),:);

%% Plot PR Curves

figure(1)
subplot(2,2,1)
plot(Scal2(:,1), Scal2(:,2), Scal1(:,1), Scal1(:,2), 'Linewidth', 2);
set(gca, 'Fontsize', 14);
subplot(2,2,2)
plot(Round2(:,1), Round2(:,2), Round1(:,1), Round1(:,2), 'Linewidth', 2);
set(gca, 'Fontsize', 14);
subplot(2,2,3)
plot(Flat2(:,1), Flat2(:,2), Flat1(:,1), Flat1(:,2), 'Linewidth', 2);
set(gca, 'Fontsize', 14);
subplot(2,2,4)
plot(Skate2(:,1), Skate2(:,2), Skate1(:,1), Skate1(:,2), 'Linewidth', 2);
leg = legend('Unprocessed Test Set', 'Preprocessed Test Set');
leg.FontSize = 16;
set(gca, 'Fontsize', 14);
xlabel('Recall (%)', 'Fontsize', 25, 'Fontweight', 'b');
ylabel('Precision (%)', 'Fontsize', 25, 'Fontweight', 'b');
