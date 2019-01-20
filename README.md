Population Disease Occurrence Models
====================================

This project generates populations according to statistics about the population using Evolutionary Computation Techniques
The technique is summarized in the paper 2019 paper: 
"Population Disease Occurrence Models Using Evolutionary Computation"
by: Olaf Dammann, Anselm Blumer, Jacob Barhak, Aaron Garrett



USAGE:
------
python PopDOM.py <InstructionsFile.yaml | PAPER>

* InstructionsFile.yaml - Optional instructions file containing the problem definitions
* PAPER - if using this key word, the program will reproduce paper results - High Performance Computing is advised after setting up a dask scheduler and dask workers
* If the user specifies no keyword, the program will look for the instructions file PopDOM.yaml to define the problem to solve


INSTALLATION & DEPENDENCIES:
----------------------------
To install:
1. Copy the files in this repository to a directory of choice - you may omit the Results.zip file
2. Install Anaconda from https://www.anaconda.com/download/
3. install Inspyred: pip install inspyred

Dependant libraries are: Inspyred, numpy, pandas, sklearn, dask, bokeh, holoviews

It is recommended you use Anaconda, yet other python environments should work as well
This code was tested on Windows 10 Python 2.7.14 and Linux 18.04 Python 2.7.15 and Anaconda (64-bit) with Inspyred 1.0.



EXAMPLES:
---------

To reproduce the results in the paper results type the following:
python PopDOM.py PAPER

To solve the problem defined in the file PopDOM.yaml
python ModelCombine.py 

To solve the problem defined in the file MyProblem.yaml
python MyProblem.yaml 


FILES:
------
PopDOM.py : Main file with calculations
CreatePlots.py : A file used to create holoviews plots when generating results to reproduce the paper
PopDOM1.yaml : A problem definitions file to define step 1 parameters in the paper
PopDOM2.yaml : A A problem definitions template file to generate step 2 parameters for different strategies used in the paper
Results.zip : Archived results created running the command python PopDOM.py PAPER
Readme.md : The file that you are reading now


VERSION HISTORY:
----------------
Development started on 7-Apr-2018 when Aaron sent the first code draft. 


DEVELOPER CONTACT INFO:
-----------------------

Please pass questions to:

Aaron Garrett
aaron.lee.garrett@gmail.com
http://sites.wofford.edu/garrettal/

Jacob Barhak Ph.D.
jacob.barhak@gmail.com
http://sites.google.com/site/jacobbarhak/


ACKNOWLEDGEMENTS:
-----------------
Thanks to Olaf Dammann and Anselm Blumer for useful discussions that led to development of this code.

LICENSE
-------

Copyright (C) 2018-2019 Jacob Barhak, Aaron Garrett
 
This file is part of the Population Disease Occurrence Models . The Population Disease Occurrence Models is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The Population Disease Occurrence Models is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.

