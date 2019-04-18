% Make images of single words and pseudowords at different contrast and
% noise levels. This is the code that has been used for MEG experiments on
% reading.
%
% Dependencies:
% vistadisp - https://github.com/vistalab/vistadisp

% lETTERS TO USE
clear
w = ['A':'P', 'R':'Z'];

% Upper or lower case?
wfun = @upper;
% wfun = @lower;

% Remove boarder?
removeboarder = 1;

%% loop over words and render
for ii = 1:length(w)
   wordIm(:,:,ii) = uint8(renderText(wfun(w(ii)),'courier',20,5));
end

if removeboarder ==1
    m = mean(wordIm,3);
    my = mean(m,2);
    mx = mean(m,1);
    ry = [min(find(my>0)) max(find(my>0))];
    rx = [min(find(mx>0)) max(find(mx>0))];
    wordIm = wordIm(ry(1):ry(2),rx(1):rx(2),:);
end

wordIm(wordIm == 0) = 127;
wordIm(wordIm == 1) = 254;


mkdir(sprintf('letterStim'));
for jj = 1:length(w)
    imwrite(wordIm(:,:,jj),fullfile('letterStim',sprintf('%s.png',w(jj))));
end
