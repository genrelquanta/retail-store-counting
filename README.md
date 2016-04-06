# retail-store-counting
Algorithms for counting number of people entering and exiting a retail store

File Structure - 

1) track.py - This file implements the command line interface for running the scripts.

2) helper.py - This file implements a function for parsing the test_data.idl file, a 
function for generating a video showing the tracking of objects and some other helper
functions used by the tracking algorithms. The countStats function uses the tracks generated 
to count the number of people entering and exiting the store. 

3) simpleTrack.py - This file implements the simple tracking algorithm

4) kalmanFilterTrack.py - This file implements the kalman filter tracking algorithm.

The script can be run using the following command:
python track.py -p path -f filename -t tracker -v -o outputDir

-p path - This is path to the directory which contains the test_data.idl file and the image files.
-f filename - This is name of the data file i.e. test_data.idl
-t tracker - This argument can take two values: simple - uses the simple algorithm for tracking, 
kalman - uses kalman tracking algorithm for tracking
-v - If this flag is specified, the tracking video is generated
-o - This is the directory in which images used for generating tracking video and the tracking video
are stored. 

eg:
python track.py -p ./test_data/ -f test_data.idl -t simple 
This will use simple algorithm to compute the counts and display results on the screen. Video isn't generated. 



