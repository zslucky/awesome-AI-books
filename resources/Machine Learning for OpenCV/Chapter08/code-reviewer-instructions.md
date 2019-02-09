Reviewer instructions
---------------------

All code for this book is maintained in Jupyter notebooks on GitHub:
https://github.com/mbeyeler/opencv-machine-learning


## File organization

All notebooks can be found here:
https://github.com/mbeyeler/opencv-machine-learning/tree/master/notebooks

They are organized by chapter and subsection.
For example, the notebooks for Chapter 2 are:
- 02.00-Working-with-Data-in-OpenCV.ipynb
- 02.01-Dealing-with-Data-Using-Python-NumPy.ipynb
- 02.02-Loading-External-Datasets-in-Python.ipynb
- 02.03-Visualizing-Data-Using-Matplotlib.ipynb
- 02.04-Visualizing-Data-from-an-External-Dataset.ipynb	
- 02.05-Dealing-with-Data-Using-the-OpenCV-TrainData-Container-in-C++.ipynb



## Running the code

There are are at least two ways to run the code:
- from within a Docker container using MyBinder,
- from within a Jupyter notebook on your local machine.



### Using MyBinder

MyBinder allows you to run Jupyter notebooks in an interactive Docker container.

First, make sure the website is currently online:
http://mybinder.org/status

Then go to:
http://mybinder.org/repo/mbeyeler/opencv-machine-learning

This will create a Docker environment that has all necessary packages installed.
Simply execute the code from within the Jupyter notebook.

If the website does not show all corresponding notebooks of the chapter,
the build is possibly outdated. In this case, please go to:
http://mybinder.org/status/mbeyeler/opencv-machine-learning
Then click on rebuild the code, then click launch.
Now everything should be up-to-date.



### Using Jupyter notebook

You basically want to follow the installation instructions in Chapter 1.

Installing stuff:
- Download and install Python Anaconda
- $ conda config --add channels conda-forge
- $ conda install opencv=3.1

Cloning the GitHub repo:
- $ git clone https://github.com/mbeyeler/opencv-machine-learning

Running Jupyter notebook:
- $ jupyter notebook
- This will open up a browser window in your current directory.
- Navigate to `opencv-machine-learning/notebooks`.
- Click on the notebook of your choice.
- Select `Kernel > Restart & Run All`.

Have fun!
