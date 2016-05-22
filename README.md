# ml-angio

1. Download roi.zip file from http://ww2.ii.uj.edu.pl/~zielinsb/?page=fmd
and unzip it in '<ml-angio-dir>/data/anonymous' directory.

2. Download imdb.zip file from http://ww2.ii.uj.edu.pl/~zielinsb/?page=fmd
and unzip it in '<ml-angio-dir>/data/anonymous' directory.

3. Download vlfeat library from http://www.vlfeat.org/
unzip it in '<ml-angio-dir>':
- to build 'vlfeat', go to '<ml-angio-dir>/vlfeat' and run make
  ensure you have MATLAB executable and mex in the path
- in Linux modify matlab.mak to link with gomp instead of iomp5
- export LD_LIBRARY_PATH=<ml-angio-dir>/vlfeat/toolbox/mex/mexa64:$LD_LIBRARY_PATH

4. To repeat the experiment call "run_expriment" function. You can analyze
successive steps of the algorithm by setting "opts.visualize" paramter on true.

5. After computations, you can analyze results using "process_results" and
"process_results_classifier_error" functions.
