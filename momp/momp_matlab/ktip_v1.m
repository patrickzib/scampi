%% Similarity matrix computation based on MPX
%% 'minlag' is optional and is set to 0 by default
%% First input time series (timeSeriesA) & subsequence length are required input arguments
%% Second input time series (timeSeriesB) is optional and it's required only for AB join
%%
function [kTriangIneqProfile] = ktip_v1(timeSeries, subseqLen, dsrate, minlag)


if nargin < 2
    error('incorrect number of input arguments');
elseif ~isvector(timeSeries)
    error('first argument must be a 1D vector');
elseif ~(isfinite(subseqLen) && floor(subseqLen) == subseqLen) || (subseqLen < 2) || (subseqLen > length(timeSeries)) 
    error('subsequence length must be an integer value between 2 and the length of the timeSeries');
end


if ~exist('minlag', 'var')
    minlag = 0;
end


n = length(timeSeries);
transposed_ = isrow(timeSeries);
if transposed_
    timeSeries = transpose(timeSeries);
end

nanmap = find(~isfinite(movsum(timeSeries, [0 subseqLen-1], 'Endpoints', 'discard')));
timeSeries(isnan(timeSeries)) = 0;
% We need to remove any NaN or inf values before computing the moving mean,
% because it uses an accumulation based method. We add additional handling
% elsewhere as needed.
mu = moving_mean(timeSeries, subseqLen);
invsig = 1./movstd(timeSeries, [0 subseqLen-1], 1, 'Endpoints', 'discard');
invsig(nanmap) = NaN;

df = [0; (1/2)*(timeSeries(1 + subseqLen : n) - timeSeries(1 : n - subseqLen))];
dg = [0; (timeSeries(1 + subseqLen : n) - mu(2 : n - subseqLen + 1)) + (timeSeries(1 : n - subseqLen) - mu(1 : n - subseqLen))];

% Output Similarity matrix
profileRowCount = n - subseqLen + 1;
profileColCount = floor(log2(dsrate));

% similarityMatrix = NaN(similarityMatrixLength, similarityMatrixLength);
curr_kTriangIneqProfile = NaN(profileRowCount, 1);
curr_kTriangIneqProfile_min = NaN(profileRowCount, 1);

kTriangIneqProfile = NaN(profileRowCount, profileColCount);
kTriangIneqProfile_min = NaN(profileRowCount, profileColCount);



% The terms row and diagonal here refer to a hankel matrix representation of a time series
% This uses normalized cross correlation as an intermediate quantity for performance reasons. 
% It is later reduced to z-normalized euclidean distance.

for diag = minlag + 1 : dsrate
    cov_ = (sum((timeSeries(diag : diag + subseqLen - 1) - mu(diag)) .* (timeSeries(1 : subseqLen) - mu(1))));
    for row = 1 : n - subseqLen - diag + 2
        col = row + diag - 1;
        cov_ = cov_ + df(row) * dg(col) + df(col) * dg(row);
        corr_ = cov_ * invsig(row) * invsig(col);
        if  isnan(curr_kTriangIneqProfile(row)) || corr_ < curr_kTriangIneqProfile(row)
            curr_kTriangIneqProfile(row) = corr_;

        end

        if isnan(curr_kTriangIneqProfile(col)) || corr_ < curr_kTriangIneqProfile(col)
            curr_kTriangIneqProfile(col) = corr_;

        end     
    end
    ktip_col = log2(diag);
    if diag > 1 && ktip_col == fix(ktip_col)
        kTriangIneqProfile(1:end, ktip_col) = curr_kTriangIneqProfile;

    end    
end


% kTriangIneqProfile = kTriangIneqProfile + kTriangIneqProfile(kTriangIneqProfileIdx);

% kTriangIneqProfile_nn = kTriangIneqProfile(kTriangIneqProfileIdx);
kTriangIneqProfile = sqrt(max(0, 2 * (subseqLen - kTriangIneqProfile), 'includenan'));



if transposed_   % matches the profile and profile index but not the motif or discord index to the input format
%     similarityMatrix = transpose(similarityMatrix);
    kTriangIneqProfile = transpose(kTriangIneqProfile);
end

end

% kTriangIneqProfile_min = 



%%
function [ res ] = moving_mean(a,w)
% moving mean over sequence a with window length w
% based on Ogita et. al, Accurate Sum and Dot Product

% A major source of rounding error is accumulated error in the mean values, so we use this to compensate. 
% While the error bound is still a function of the conditioning of a very long dot product, we have observed 
% a reduction of 3 - 4 digits lost to numerical roundoff when compared to older solutions.

res = zeros(length(a) - w + 1, 1);
p = a(1);
s = 0;

for i = 2 : w
    x = p + a(i);
    z = x - p;
    s = s + ((p - (x - z)) + (a(i) - z));
    p = x;
end

res(1) = p + s;

for i = w + 1 : length(a)
    x = p - a(i - w);
    z = x - p;
    s = s + ((p - (x - z)) - (a(i - w) + z));
    p = x;
    
    x = p + a(i);
    z = x - p;
    s = s + ((p - (x - z)) + (a(i) - z));
    p = x;
    
    res(i - w + 1) = p + s;
end

res = res ./ w;

end
