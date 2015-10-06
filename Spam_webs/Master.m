%% Spam Classification with SVMs
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processWebsite.m
%     websiteFeatures.m
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Website Preprocessing ====================
%  To use an SVM to classify websites into Spam v.s. Non-Spam, you first need
%  to convert each website into a vector of features. In this part, you will
%  implement the preprocessing steps for each website. You should
%  complete the code in processEmail.m to produce a word indices vector
%  for a given website.

fprintf('\nPreprocessing sample website (websiteSample1.txt)\n');

% Extract Features
file_contents = readFile('websiteSample1.txt');
word_indices  = processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================
%  Now, you will convert each website into a vector of features in R^n. 
%  You should complete the code in websiteFeatures.m to produce a feature
%  vector for a given website.

fprintf('\nExtracting features from sample website (websiteSample1.txt)\n');

% Extract Features
file_contents = readFile('websiteSample1.txt');
word_indices  = processEmail(file_contents);
features      = websiteFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Train Linear SVM for Spam Classification ========
%  In this section, you will train a linear classifier to determine if an
%  website is Spam or Not-Spam.

% Load the Spam Website dataset
% You will have X, y in your environment
load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment
load('train_v2.CSV');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;


%% ================= Part 5: Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an website is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
%

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Part 6: Try Your Own Websites =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  websites! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, websiteSample1.txt and websiteSample2.txt as examples. 
%  The following code reads in one of these websites and then uses your 
%  learned SVM classifier to determine whether the website is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% websiteSample1.txt or websiteSample2.txt to see different predictions on
% different websites types). Try your own websites as well!
filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = websiteFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

