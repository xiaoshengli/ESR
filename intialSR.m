function SR = intialSR(TRAIN, NP, K, L)
% Initialize the SRs using randomly selected subsequences in the dataset

seriesLength = size(TRAIN, 2);
SR = zeros(NP, K*sum(ceil(L*seriesLength)));
randSeries = unidrnd(size(TRAIN, 1), NP, K*length(L));
L2 = repmat(L, K, 1);
L2 = reshape(L2, 1, length(L)*K);
insertPoint = cumsum(ceil(L2*seriesLength)) - ceil(L2*seriesLength) + 1;

for i = 1 : length(L)*K
    series = TRAIN(randSeries(:, i), :);
    len = ceil(L(floor((i-1)/K)+1)*seriesLength);
    startPoint = unidrnd(seriesLength-len+1, NP, 1);
    selectedSub = ones(len+1, NP);
    selectedSub(1, :) = startPoint';
    selectedSub = cumsum(selectedSub);
    selectedSub = selectedSub(1:size(selectedSub, 1)-1, :);
    selectedSub = selectedSub';
    indexAdjust = (1:NP)';
    selectedSub = (selectedSub-1) * NP + indexAdjust(:, ones(1, len));
    SR(:, insertPoint(i) : insertPoint(i)+len-1) = series(selectedSub);
end
