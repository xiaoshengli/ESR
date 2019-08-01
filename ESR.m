%Copyright (c) 2017 Xiaosheng Li
%Email: xli22@gmu.edu
%Reference: Evolving Separating References for Time Series Data Mining

clc;
clear;

%These are the 20 smallest becnmarks used in the paper
benchmarks = {'Beef', 'CBF', 'Coffee','DiatomSizeReduction', ...
    'ECGFiveDays', 'FaceFour', 'FacesUCR', 'Gun_Point', 'ItalyPowerDemand',...
    'Lighting2', 'Lighting7', 'MedicalImages', 'MoteStrain', 'OliveOil', ...
    'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'Symbols',...
    'synthetic_control', 'Trace', 'TwoLeadECG'};

%The user needs to specify the dataset to run on and the binary parameter,
%parameter 1 corresponds to "S1" in the paper and 2 corresponds to "S2".

dataset = 'SonyAIBORobotSurface';
parameter = 1;

datapath = strcat('./UCR_TS_Archive_2015/', dataset, '/', dataset);

%The data is already normalized, so it is directly used here.
TRAIN = load(strcat(datapath, '_TRAIN')); 
TEST = load(strcat(datapath, '_TEST'));

%Separate the time series and labels
TRAIN_class_labels = TRAIN(:, 1);
TRAIN(:, 1) = [];

TEST_class_labels = TEST(:, 1);
TEST(:, 1) = [];

%Count the number of classes
ClassList = unique(TRAIN_class_labels);
C = length(ClassList);

% The population size
NP = 40;

% Set the distance formula, K and L
switch parameter
    case 1
        normalization = 0;
        L = [0.05 0.1 0.2 0.4 0.6];
        K = 5;
    case 2
        normalization = 1;
        L = [0.2 0.4 0.6];
        K = 5;
end

% Display the dataset name and parameter used
disp(['dataset: ' dataset]);
disp(['parameter: (S)' num2str(parameter)]);

%Caching to speedup the distance computation
if parameter == 1
    cum_sumx = [];
    cum_sumx2 = [];
    cum_sumx_t = [];
    cum_sumx2_t = [];
else
    cum_sumx = cumsum(TRAIN, 2);
    cum_sumx2 = cumsum(TRAIN.^2, 2);
    cum_sumx_t = cumsum(TEST, 2);
    cum_sumx2_t = cumsum(TEST.^2, 2);
    cum_sumx = [zeros(size(TRAIN, 1), 1) cum_sumx];
    cum_sumx2 = [zeros(size(TRAIN, 1), 1) cum_sumx2];
    cum_sumx_t = [zeros(size(TEST, 1), 1) cum_sumx_t];
    cum_sumx2_t = [zeros(size(TEST, 1), 1) cum_sumx2_t];
end

% SRLength corresponds to P in the paper
seriesLength = size(TRAIN, 2);
n = K*sum(ceil(L*seriesLength)) + K*length(L)+2;
L2 = repmat(L, K, 1);
L2 = reshape(L2, 1, length(L)*K);
SRLength = ceil(L2*seriesLength);

%Set the lower bounds and upper bounds
minTRAIN =  min(min(TRAIN));
maxTRAIN = max(max(TRAIN));

alpha = 2;
lowerBound = (minTRAIN<=0)*alpha*minTRAIN + (minTRAIN>0)*minTRAIN/alpha;
upperBound = (maxTRAIN>=0)*alpha*maxTRAIN + (maxTRAIN<0)*maxTRAIN/alpha;

lowerSR = lowerBound*ones(NP, K*sum(ceil(L*seriesLength)));
upperSR = upperBound*ones(NP, K*sum(ceil(L*seriesLength)));

lowerBoundXI = 0;
upperBoundXI = 0.1;
lowerXI = lowerBoundXI*ones(NP, 1);
upperXI = upperBoundXI*ones(NP, 1);

lowerW = -inf(NP, K*length(L)+1);
upperW = inf(NP, K*length(L)+1);

LB = [lowerSR lowerXI lowerW];
UB = [upperSR upperXI upperW];

%Set variables to record the best individuals and the respective F values
bestX = zeros(C, n);
bestFit = zeros(C, 2);

%Parallelize the evaluation of individuals in the population
% delete(gcp('nocreate'));
% parpool(4);

% Train for each class in the dataset
for c = 1 : C 
    
    rng(100*c);
        
    disp(['Training for class ',num2str(c),' ...']);
    
    % Transform the labels into binary values
    Y = TRAIN_class_labels;
    Pos = Y == ClassList(c);
    Neg = ~Pos;
    Y(Pos) = 1;
    Y(Neg) = -1;
    
    % Initialize the SRs using randomly selected subsequences in the
    % dataset
    SR = intialSR(TRAIN, NP, K, L);
    
    % Intitialize the decison boudary
    W = randn(NP, K*length(L)+1);
    
    XI = lowerXI + rand(NP, 1) .* (upperXI - lowerXI);
    
    % X is the population of individuals
    X = [SR XI W];
    
    %Set the initial generation to 1, maxGen is the total generation to
    %perform
    gen = 1;
    maxGen = 5000;
    
    %Evaluate the individuals, FX is the f(x) values in paper and gX is the
    %G(x) values in the paper
    [FX, gX] = evaluation(X, TRAIN, Y, K, L, SRLength, cum_sumx, cum_sumx2, normalization);

    while gen <= maxGen

        F = [1.0 1.0 0.8];
        CR = [0.1 0.9 0.2];
        paraIndex = floor(rand(NP, 3) * length(F)) + 1;

        %Mutation operation: rand/1 strategy
        [r1, r2, r3] = select3rands_(NP);
        tempF = (F(paraIndex(:, 1)))';
        tempCR = (CR(paraIndex(:, 1)))';
        V = X(r1, :) + tempF(:,ones(1, n)).*(X(r2, :)-X(r3, :));
        
        %Handle boundary constraint violation
        BL = V<LB; 
        V(BL) = 2*LB(BL) - V(BL);
        BLU = V(BL)>UB(BL); 
        BL(BL) = BLU; 
        V(BL) = UB(BL);
        BU = V>UB; 
        V(BU) = 2*UB(BU) - V(BU);
        BUL = V(BU)<LB(BU); 
        BU(BU) = BUL; 
        V(BU) = LB(BU);
        
        %Crossover operation
        J_= mod(floor(rand(NP, 1)*n), n) + 1;
        J = (J_-1)*NP + (1:NP)';
        crs = rand(NP, n) < tempCR(:, ones(1, n));

        U1 = X;
        U1(J) = V(J);
        U1(crs) = V(crs);

        % current to rand/1 strategy
        muIndex = floor(rand(NP, 3) * NP) + 1;
        r1 = muIndex(:, 1);
        r2 = muIndex(:, 2);
        r3 = muIndex(:, 3);
        tempF = (F(paraIndex(:, 2)))';
        tempRand = rand(NP, 1);
        V = X + tempRand(:, ones(1,n)) .* (X(r1, :) - X) + tempF(:, ones(1, n)).*(X(r2, :)-X(r3, :));
        
        %Handle boundary constraint violation
        BL = V<LB; 
        V(BL) = 2*LB(BL) - V(BL);
        BLU = V(BL)>UB(BL); 
        BL(BL) = BLU; 
        V(BL) = UB(BL);
        BU = V>UB; 
        V(BU) = 2*UB(BU) - V(BU);
        BUL = V(BU)<LB(BU); 
        BU(BU) = BUL; 
        V(BU) = LB(BU);

        U2 = V;

        % rand/2 strategy
        [r1, r2, r3, r4, r5] = select5rands_(NP);
        tempF = (F(paraIndex(:, 3)))';
        tempCR = (CR(paraIndex(:, 3)))';
        tempRand = rand(NP, 1);
        V = X(r1, :) + tempRand(:, ones(1, n)) .* (X(r2, :)-X(r3, :)) + tempF(:, ones(1, n)).*(X(r4, :)-X(r5, :));
        
        %Handle boundary constraint violation
        BL = V<LB; 
        V(BL) = 2*LB(BL) - V(BL);
        BLU = V(BL)>UB(BL); 
        BL(BL) = BLU; 
        V(BL) = UB(BL);
        BU = V>UB; 
        V(BU) = 2*UB(BU) - V(BU);
        BUL = V(BU)<LB(BU); 
        BU(BU) = BUL; 
        V(BU) = LB(BU);
        
        %Crossover operation
        J_= mod(floor(rand(NP, 1)*n), n) + 1;
        J = (J_-1)*NP + (1:NP)';
        crs = rand(NP, n) < tempCR(:, ones(1, n));

        U3 = X;
        U3(J) = V(J);
        U3(crs) = V(crs);
        
        %U is the newly generated trial vectors
        U = [U1; U2; U3];
        
        %Evaluate the trial individuals
        [FU, gU] = evaluation(U, TRAIN, Y, K, L, SRLength, cum_sumx, cum_sumx2, normalization);

        F = [FX; FU];
        g = [gX; gU];
        
        %fit is the varaible containing f(x) and G(x)
        fit = [F g];
        
        %Selection operation
        % The population contains no feasible solution
        if fit(:, 2) > 0
            
            % Choose the individuals for the next population according to
            % G(x) values
            fitX = fit(1:NP, :);
            fitU1 = fit(NP+1:2*NP, :);
            fitU2 = fit(2*NP+1:3*NP, :);
            fitU3 = fit(3*NP+1:4*NP, :);

            gX = g(1:NP, :);
            gU1 = g(NP+1:2*NP, :);
            gU2 = g(2*NP+1:3*NP, :);
            gU3 = g(3*NP+1:4*NP, :);

            S = fitU1(:, 2) <= fitX(:, 2);
            X(S, :) = U1(S, :);
            fitX(S, :) = fitU1(S, :);
            gX(S, :) = gU1(S, :);

            S = fitU2(:, 2) <= fitX(:, 2);
            X(S, :) = U2(S, :);
            fitX(S, :) = fitU2(S, :);
            gX(S, :) = gU2(S, :);

            S = fitU3(:, 2) <= fitX(:, 2);
            X(S, :) = U3(S, :);
            fitX(S, :) = fitU3(S, :);
            gX(S, :) = gU3(S, :);

            FX = fitX(:, 1);

        % The population contains some feasible solutions
        else

                % fFun denotes f(x) function values
                fFun = fit(:, 1);

                % conVio denotes the degree of constraint violation G(x)
                conVio = fit(:, 2);

                %Normalize the f(x) values
                norfFun = (fFun - min(fFun))./((max(fFun) - min(fFun)) + 1E-30);

                %Normalize the G(x) values
                norConVio = (conVio - min(conVio))./((max(conVio) - min(conVio)) + 1E-30);

                % Record the subscripts of the feasible solutions
                S1 = find(fit(:, 2) == 0);

                % Record the subscripts of the infeasible solutions
                S2 = find(fit(:, 2) > 0);

                % Compute the minimum normalized f(x) function values
                % of the feasible individuals and infeasible individuals
                minfFunS1 = min(norfFun(S1));
                minfFunS2 = min(norfFun(S2));

            % The minimum normalized f(x) values of the feasible
            % individuals is larger than that of the infeasible individuals
            if minfFunS2 < minfFunS1

                norfFunS2 = norfFun(S2);
                norConVioS2 = norConVio(S2);
                S3 = find(norfFunS2 < minfFunS1);

                % Compute the minimum penalty coefficient               
                r = 1.0001 * max((minfFunS1 - norfFunS2(S3)) ./ (norConVioS2(S3)));

                % Obtain the final evaluation function values F(x)
                finalFit = norfFun + r * norConVio;

                % Choose the individuals for the next population according
                % to F(x)
                fitX = fit(1:NP, :);
                fitU1 = fit(NP+1:2*NP, :);
                fitU2 = fit(2*NP+1:3*NP, :);
                fitU3 = fit(3*NP+1:4*NP, :);

                gX = g(1:NP, :);
                gU1 = g(NP+1:2*NP, :);
                gU2 = g(2*NP+1:3*NP, :);
                gU3 = g(3*NP+1:4*NP, :);

                finalFitX = finalFit(1:NP, :);
                finalFitU1 = finalFit(NP+1:2*NP, :);
                finalFitU2 = finalFit(2*NP+1:3*NP, :);
                finalFitU3 = finalFit(3*NP+1:4*NP, :);

                S = finalFitU1 <= finalFitX;
                X(S, :) = U1(S, :);
                fitX(S, :) = fitU1(S, :);
                finalFitX(S, :) = finalFitU1(S, :);
                gX(S, :) = gU1(S, :);

                S = finalFitU2 <= finalFitX;
                X(S, :) = U2(S, :);
                fitX(S, :) = fitU2(S, :);
                finalFitX(S, :) = finalFitU2(S, :);
                gX(S, :) = gU2(S, :);

                S = finalFitU3 <= finalFitX;
                X(S, :) = U3(S, :);
                fitX(S, :) = fitU3(S, :);
                finalFitX(S, :) = finalFitU3(S, :);
                gX(S, :) = gU3(S, :);

                FX = fitX(:, 1);

            % The minimum normalized f(x) function values of the feasible
            % solutions is not larger than that of the infeasible solutions
            else

            % Choose the individuals for the next population
            % Under this condition, the minimum penalty coefficient rmin is equal to zero.                
            % We only need to consider the f(x) values

                fitX = fit(1:NP, :);
                fitU1 = fit(NP+1:2*NP, :);
                fitU2 = fit(2*NP+1:3*NP, :);
                fitU3 = fit(3*NP+1:4*NP, :);

                gX = g(1:NP, :);
                gU1 = g(NP+1:2*NP, :);
                gU2 = g(2*NP+1:3*NP, :);
                gU3 = g(3*NP+1:4*NP, :);

                S = fitU1(:, 1) <= fitX(:, 1);
                X(S, :) = U1(S, :);
                fitX(S, :) = fitU1(S, :);
                gX(S, :) = gU1(S, :);

                S = fitU2(:, 1) <= fitX(:, 1);
                X(S, :) = U2(S, :);
                fitX(S, :) = fitU2(S, :);
                gX(S, :) = gU2(S, :);

                S = fitU3(:, 1) <= fitX(:, 1);
                X(S, :) = U3(S, :);
                fitX(S, :) = fitU3(S, :);
                gX(S, :) = gU3(S, :);

                FX = fitX(:, 1);
            end
        end

        if mod(gen, 1000) == 0
            disp(['The evolution generation reaches ',num2str(gen),' ...']);
        end
        
        % If there is no feasible individual in 5000th generation, extend
        % maxGen to 10000
        if (gen == 5000) && (sum(fitX(:,2)==0) == 0)
            maxGen = 10000;
        end
        
        % Update the generation number
        gen = gen + 1;
        
    end

    % Record the best individual and the evaluation value
    finX = X;
    finFit = fitX;
    [sortedFit, index] = sortrows(finFit, [2 1]);
    bestX(c, :) = finX(index(1), :);
    bestFit(c, :) = sortedFit(1, :);
    
end

%Extract the SRs and decision boundary
SR = bestX(:, 1 : K*sum(ceil(L*seriesLength)));
w = bestX(:, K*sum(ceil(L*seriesLength))+2 : end);

% Classify the testing time series with the SRs and decision boundary
h = SRClassify(SR, w, TEST, SRLength, ClassList, C, cum_sumx_t, cum_sumx2_t, normalization);

% Calculate and output the testing error rate
errorRate = sum(h~=TEST_class_labels)/length(TEST_class_labels);
disp(['The testing error rate is ',num2str(errorRate)])

% delete(gcp('nocreate'));