fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data1.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');


fprintf('Solving with normal equations...\n');


%% Load Data
data = csvread('ex1data1.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
predict=X*theta
predict1=[1,6.2,180,0.441,0.775]*theta
fprintf('For given valuse,prediction of points/game is: %f\n',...
    predict1);
fprintf('\nTraining Set Accuracy: %f\n', mean(double((predict- y<=2 & predict-y>=0)|(predict- y<=0 & predict-y>=-2) )) * 100);