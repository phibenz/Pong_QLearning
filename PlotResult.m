F1='Archive/01_161211_40/LearningFile/';
L1=importdata([F1, 'learning1.csv']);
R1=importdata([F1, 'reward1.csv']);

F2='Archive/02_161211_4040/LearningFile/';
L2=importdata([F2, 'learning1.csv']);
R2=importdata([F2, 'reward1.csv']);

F3='Archive/03_161211_404040/LearningFile/';
L3=importdata([F3, 'learning1.csv']);
R3=importdata([F3, 'reward1.csv']);

F4='Archive/04_161211_40404040/LearningFile/';
L4=importdata([F4, 'learning1.csv']);
R4=importdata([F4, 'reward1.csv']);

figure(1)
clf 
hold on
plot(L1.data(1:400,2), L1.data(1:400,1), '-k')
plot(L2.data(1:400,2), L2.data(1:400,1), '--k')
plot(L3.data(1:400,2), L3.data(1:400,1), ':k')
plot(L4.data(1:400,2), L4.data(1:400,1), '-.k')
grid on
axis([0 400 0 1])
%ax=gca;
ylabel('Loss', 'Interpreter', 'Latex')
xlabel('Epochs', 'Interpreter', 'Latex')
l=legend('1 hidden layer', ...
       '2 hidden layers', ...
       '3 hidden layers', ...
       '4 hidden layers', ...
       'Location', 'northwest');
set(l,'Interpreter','latex')

figure(2)
clf
hold on
plot(R1.data(:,1), R1.data(:,2), '-k')
plot(R2.data(:,1), R2.data(:,2), '--k')
plot(R3.data(:,1), R3.data(:,2), ':k')
plot(R4.data(:,1), R4.data(:,2), '-.k')
ylabel('Reward', 'Interpreter', 'Latex')
grid on
l=legend('1 hidden layer', ...
       '2 hidden layers', ...
       '3 hidden layers', ...
       '4 hidden layers', ...
       'Location', 'northwest');
set(l,'Interpreter','latex')