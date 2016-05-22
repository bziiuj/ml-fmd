run vlfeat/toolbox/vl_setup

root = fileparts(mfilename('fullpath')) ;

addpath(fullfile(root, 'vlfeat/toolbox')) ;
addpath(fullfile(root, 'vlfeat/toolbox/misc')) ;
addpath(fullfile(root, 'vlfeat/toolbox/mex')) ;
if ismac
  addpath(fullfile(root, 'vlfeat/toolbox/mex/mexmaci64')) ;
elseif isunix
  addpath(fullfile(root, 'vlfeat/toolbox/mex/mexa64')) ;
end
