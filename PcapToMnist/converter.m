% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program reads raw MNIST files available at 
% http://yann.lecun.com/exdb/mnist/ 
% and converts them to files in matlab format 
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.  

% This program was originally written by HBQ

% Work with test files first 
fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n'); 

f = fopen('t10k-images-idx3-ubyte','r');
magicf = fread(f, 1, 'int32', 0, 'ieee-be');          
numImagesf = fread(f, 1, 'int32', 0, 'ieee-be');
numRowsf = fread(f, 1, 'int32', 0, 'ieee-be'); 
numColsf = fread(f, 1, 'int32', 0, 'ieee-be');
g = fopen('t10k-labels-idx1-ubyte','r');
magic = fread(g,1,'int32',0,'ieee-be');
numLabels = fread(g,1,'int32',0,'ieee-be');
fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n'); 
  
  rawimages = fread(f,Inf,'unsigned char');
  y = fread(g,Inf,'unsigned char');
  x = reshape(rawimages,28,28,numImagesf);
  
  save('compare_test','x','y');
  
  f = fopen('train-images-idx3-ubyte','r');
magicf = fread(f, 1, 'int32', 0, 'ieee-be');          
numImagesf = fread(f, 1, 'int32', 0, 'ieee-be');
numRowsf = fread(f, 1, 'int32', 0, 'ieee-be'); 
numColsf = fread(f, 1, 'int32', 0, 'ieee-be');
g = fopen('train-labels-idx1-ubyte','r');
magic = fread(g,1,'int32',0,'ieee-be');
numLabels = fread(g,1,'int32',0,'ieee-be');
fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n'); 

  
  rawimages = fread(f,Inf,'unsigned char');
  y = fread(g,Inf,'unsigned char');
  x = reshape(rawimages,28,28,numImagesf);
  
  save('compare_train','x','y');
  
  



