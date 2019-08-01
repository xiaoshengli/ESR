function [f, g] = evaluation(X, TRAIN, Y, K, L, SRLength, cum_sumx, cum_sumx2, normalization)
%Calcuate the f(x) and G(x) values of the individuals

seriesLength = size(TRAIN, 2);

%Extract the SRs and decision boundaries
SR = X(:, 1 : K*sum(ceil(L*seriesLength)));
XI = X(:, K*sum(ceil(L*seriesLength))+1);
W = X(:, K*sum(ceil(L*seriesLength))+2 : end);

% Calculate G(x) values
g = (arrayfun(@(x) constraintViolation(SR(x, :),XI(x, :), W(x, :), TRAIN, Y, ...
    SRLength, cum_sumx, cum_sumx2, normalization), 1:size(X, 1)))';

%Parallelize the evaluation of individuals in the population
% g = zeros(size(X, 1), 1);
% parfor i = 1 : size(X, 1)
%     g(i) = constraintViolation(SR(i,:), W(i, :), TRAIN, Y, SRLength, cum_sumx, cum_sumx2, normalization);
% end

%Calcuate f(x) values
C = 10000;
f = sqrt(sum(W(:,2:end).^2, 2)) + C*abs(XI); 