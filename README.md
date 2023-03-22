# CSC2012_Project

# Why use conda?

For tensorflow pip install on m1 mac will have issue. So we use conda to install tensorflow.


# Step 1: Install Conda

To install Conda, you can follow the instructions on the [official website](https://conda.io/en/latest/miniconda.html). 


# Step 2: Update Conda

Once you have installed Conda, you should update it to the latest version using the following command:

```
conda update conda
```

# Step 3: Create a Conda environment

A Conda environment is a virtual environment where you can install packages without affecting the rest of your system. To create a new environment, run the following command:

```
conda env create -n myenv -f environment.yml

OR

conda env create -n myenv -f environment-silicon-mac.yml
```

# Step 4 : Activate the environment

To activate the environment, run the following command:

```
conda activate csc2012app
```
