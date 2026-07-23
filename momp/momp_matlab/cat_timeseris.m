function [aligned_T, split] = cat_timeseris(T1, T2, dd)

if isrow(T1), T1 = T1'; end
if isrow(T2), T2 = T2'; end

if isempty(T1), aligned_T = T2; split = nan;
elseif isempty(T2), aligned_T = T1; split = nan;
        
else

    if exist('dd','var') && dd > 1
        n = length(T1);
        pad = (ceil(n/dd)*dd) - n;
        T1 = [T1 ; randn(pad,1)];
        [T1, ~] = paa(T1, floor(n/dd));
        n = length(T2);
        pad = (ceil(n/dd)*dd) - n;
        T2 = [T2 ; randn(pad,1)];
        [T2, ~] = paa(T2, floor(n/dd));
    end

    startpoint = T2(1);
    endpoint= T1(end);

    shiftval = endpoint - startpoint;
    aligned_T = [T1 ; T2 + shiftval];
    split = length(T1)+1;
end

end