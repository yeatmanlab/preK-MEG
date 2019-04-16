% Example 1: Synthesis of a "text" texture image, using
% Portilla-Simoncelli texture analysis/synthesis code, based on
% alternate projections onto statistical constraints in a complex
% overcomplete wavelet representation.
%
% See Readme.txt, and headers of textureAnalysis.m and
% textureSynthesis.m for more details.
%
% Javier Portilla (javier@decsai.ugr.es).  March, 2001

imdir = '~/git/SSWEF/stim/word_upper_c254_p0/';
outdir = '~/git/SSWEF/stim/word_upper_c254_p0_portilla/';
if ~exist(outdir,'dir'),mkdir(outdir);,end

imList = dir(fullfile(imdir, '*.png'));

for ii = 1:length(imList)
    fprintf('synthesizing %s\n',imList(ii).name)
    imRaw = imread(fullfile(imdir,imList(ii).name));
    im0 = double(imresize(imRaw,[128 256]));	% im0 is a double float matrix!
    
    Nsc = 4; % Number of scales
    Nor = 4; % Number of orientations
    Na = 9;  % Spatial neighborhood is Na x Na coefficients
    % It must be an odd number!
    
    params = textureAnalysis(im0, Nsc, Nor, Na);
    
    Niter = 500;	% Number of iterations of synthesis loop
    Nsx = 256;	% Size of synthetic image is Nsy x Nsx
    Nsy = 128;	% WARNING: Both dimensions must be multiple of 2^(Nsc+2)
    
    res = textureSynthesis(params, [Nsy Nsx], Niter);
    
    resIm=uint8(imresize(res, size(imRaw)));
    
    % Threshold image
    resIm(resIm<200) = 127; resIm(resIm > 127) = max(imRaw(:));
    %figure;
    %subplot(2,2,1);title('Original');imshow(imRaw);subplot(2,2,2);title('Synthesized');imshow(resIm);
    %subplot(2,2,3);imhist(imRaw(:));subplot(2,2,4);imhist(resIm(:));
    
    % Write
    imwrite(resIm,fullfile(outdir,imList(ii).name));
end

