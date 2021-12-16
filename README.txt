HANDS-ON TUTORIAL for Multitask neural networks

In the following sections we present an hands-on tutorial for creating your own multitask neural network. After describing the required software and data we will show i) how a traditional multitask problem is formulated, ii) how to check the task-relatedness assumptions, iii) how to build simple single and multitask fully-connected neural networks and iv) how to compare the obtained results. We will use nuclear receptor modulation as a toxicological case study.

Getting started

All calculations were performed using Python 3.9.7. As a framework for Python, we utilized Jupyter Notebooks, an interactive open-source web-based application (https://jupyter.org/).
With the provided notebook, users can reproduce the worked example by executing one block of code at a time. The implemented chemical language models rely on Keras with Tensorflow backend, an open-source machine learning library for deep learning. 

Preparatory step
As a prerequisite for running the program code for multitask learning, the following software must be installed locally:

Anaconda, an open-source distribution of Python programming language for scientific computing, that simplifies package management and deployment. Anaconda for Python 3.9 can be installed from the official webpage (URL: www.anaconda.com).

(optional) Git, a version control system. Git installation is platform dependent; refer to the instructions available at www.atlassian.com/git/tutorials/install-git.

Code Download
The open-source code is available as a GitHub repository at URL: github.com/CValsecchi.... Users can clone the repository (i.e., obtain a local copy of the repository contents) with the following command on a Linux/Mac terminal or Windows command line:

git clone https://github.com/CValsecchi…

A copy of the repository will be generated on the local computer in the dedicated GitHub folder. It is also possible to manually download a compressed file of the repository by selecting the download button in the GitHub repository. You can then move to the repository, e.g., with the following commands:

cd <path/to/folder> (from a Mac or Linux terminal)
cd <path\to\folder> (from Windows command line)

To identify the directory where cloned repositories are stored on the local computer.

A virtual environment containing all the necessary packages is created using the provided “environment.yml” file, as follows:

conda env create -f environment.yml

The installation might take some time. Once the environment is set up, it is activated using the following command:

conda activate MTL_env

To use the provided notebook, go to the example folder and launch the Jupyter Notebook application from your terminal or command line, as follows:

jupyter notebook

A web page will open, showing the content of the code folder. Double clicking on the file named “Multitask_learning_for_SAR.ipynb” opens the notebook. Each line of the provided code can be executed in a stepwise fashion to visualize and reproduce the results presented in this chapter. The software was tested using an Intel® CoreTM i7-6950X CPU processor with a dedicated RAM of 32 GB.

Happy coding!
