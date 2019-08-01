function g = constraintViolation(SR, xi, w, TRAIN, Y, SRLength, cum_sumx, cum_sumx2, normalization)

%Transform time series into new space instances using the SR
%mex functions are used to accelerate the computation
%individualDistance_c is the function calculates the non-normalized distance 
%normalizedIndividualDistance_c is the function calculates the normalized distance 

if normalization == 0
    z = individualDistance_c(TRAIN, SR, SRLength);
else
    z = normalizedIndividualDistance_c(TRAIN, SR, SRLength, cum_sumx, cum_sumx2);
end

%Calcuate G(x) values
h = z * (w(2 : end))' + w(1);
g = 1 - Y .* h - xi;
g = sum(max(g, 0));