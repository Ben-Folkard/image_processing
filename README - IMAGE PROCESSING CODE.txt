README - IMAGE PROCESSING CODE
ELLA BECK
LAST MODIFIED 07/10/25


hi, this should be everything you need to know to start using the code :)
if any problems or questions arise feel free to contact me!
will be making a few amendments to comments/readability in the next few weeks, nothing profound and nothing that will edit any functionality
i apologise in advance about the efficiency (and the linting)

FILE DESCRIPTIONS
image_processing.ipynb
a jupyter notebook, used to play around with the main module, generate images, observe various metrics from video snippets

image_processing_flowchart.png
used in a poster, details basic code logic of main module (NOTE: THIS IS OUTDATED! will update soon, some functions have been reordered since for efficiency)

image_processing_optimisation.py
the main module (i will rename this!! as it is far from optimised lol, just a relic from when i first started building the code)

image_processing_optimisation_script.sh
the shell script i used with scarf, all parameters should be fine as they are but adjust if desired

test_image_processing_optimisation.py
contains all my unit tests, coverage isn't exactly extensive but most functions have basic tests implemented

NOTES:
this code is far from perfect, i had a lot of trouble identifying damaged pixels in instances where the camera moved, so there exist a few weird methods i used to try and combat this (including discarding frames entirely) - if this is ever fixed properly i would love to know how!

this was tested almost exclusively with the video footage '11_01_H_170726081325.avi', but should work the same on all robot vids

you will discover very quickly that damaged pixels do not have the brightness/contrast/collimation that you would expect, i have employed a lot of weird methods to try and distinguish them, but their features are very similar to tiny reflections from metals and other surfaces in the video. i could never find a surefire way to exclude all of them for certain so the uncertainty on my results will be very high

interestingly the way the camera is facing seems to have an effect on the number of damaged pixels?

also be very careful in changing function order/other processes for the sake of efficiency - the logic of the code is quite intricate in places and it would be easy to disturb and make the code not work quite right (sorry), so be conscious of this and use as many unit tests as you can