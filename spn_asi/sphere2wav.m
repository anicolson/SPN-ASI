%% REWRITES TIMIT FILES TO REMOVE SPHERE-BASED ERRORS.
clear all; close all; clc;
train = dir('/home/aaron/datasets/timit_wav/timit/train/*/*/*.wav');
test = dir('/home/aaron/datasets/timit_wav/timit/test/*/*/*.wav');
for i = 1:length(train)
    [x, fs] = audioread(strcat(train(i).folder, '/', train(i).name));
    audiowrite(strcat(train(i).folder, '/', train(i).name), x, fs);
    clc;
    fprintf('%i of %i complete.\n',i,length(train));
end

for i = 1:length(test)
    [x, fs] = audioread(strcat(test(i).folder, '/', test(i).name));
    audiowrite(strcat(test(i).folder, '/', test(i).name), x, fs);
    clc;
    fprintf('%i of %i complete.\n',i,length(test));
end
