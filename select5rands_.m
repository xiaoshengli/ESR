function [r1,r2,r3,r4,r5] = select5rands_(NP)

r1 = zeros(NP,1);
r2 = zeros(NP,1);
r3 = zeros(NP,1);
r4 = zeros(NP,1);
r5 = zeros(NP,1);

for i = 1:NP
    nouse = 1:NP;
    nouse(i) = [];
    id = floor(rand*(NP-1))+1;
    r1(i) = nouse(id);
    nouse(id) = [];
    id = floor(rand*(NP-2))+1;
    r2(i) = nouse(id);
    nouse(id) = [];
    id = floor(rand*(NP-3))+1;
    r3(i) = nouse(id);
    nouse(id) = [];
    id = floor(rand*(NP-4))+1;
    r4(i) = nouse(id);
    nouse(id) = [];
    r5(i) = nouse(floor(rand*(NP-5))+1);
end