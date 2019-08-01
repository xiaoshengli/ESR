function h = SRClassify(SR, w, TEST, SRLength, ClassList, C, cum_sumx_t, cum_sumx2_t, normalization)
% Classify the testing time series with the SRs and decision boundary

H = zeros(size(TEST, 1), C);

for k = 1 : C
    if normalization == 0
        z = individualDistance_c(TEST, SR(k, :), SRLength);
    else
        z = normalizedIndividualDistance_c(TEST, SR(k, :), SRLength, cum_sumx_t, cum_sumx2_t);
    end
    H(:, k) = z * (w(k, 2 : end))' + w(k,1);
end

% h is the predicted label
[~, I] = max(H, [], 2);
h = ClassList(I);