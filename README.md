# CSC2012_Project

# Setting up Poetry

Poetry is a dependency manager and a build tool for Python. This guide will walk you  
through the steps required to set up Poetry on both macOS and Windows.
<br/>

## Prerequisites
Before we get started, you need to make sure that you have the following tools installed
on your computer:
- Python 3.6 or higher
- pip (the Python package manager)
- Homebrew package manager (for mac user)

## Installation of Poetry

### macOS
1. Install `pipx` using Homebrew:
    ```zsh
    brew install pipx
    ```
2. Ensure that `pipx` is in your system's PATH using the following command:
    ```zsh
    pipx ensurepath
    ```

3. Install Poetry using `pipx`:
    ```zsh
    pip install poetry
    ```
4. Once the installation is complete, run the following command to configure Poetry to
create the virtual environment within your project directory:
    ```zsh
    poetry config virtualenvs.in-project true
    ```

5. Create a new virtual environment using Poetry:
    ```commandline
    poetry env use
    ```
    This will create a new virtual environment in your project directory.

7. Verify that Poetry is installed correctly by running the following command:

    ```zsh
    poetry --version
    ```

### Windows

1. Use pip to install Poetry by running the following command
    ```commandline
    pip install poetry
    ```

2. Once the installation is complete, run the following command to configure Poetry to
create the virtual environment within your project directory:
    ```commandline
    poetry config virtualenvs.in-project true
    ```
3. Create a new virtual environment using Poetry:
    ```commandline
    poetry env use
    ```
    This will create a new virtual environment in your project directory.

## Installation of Package Dependencies

1. Open the terminal or command prompt.
2. Navigate to the directory of your project that contains the `pyproject.toml` and `poetry.lock` file.
3. Run the following command to install the package dependencies for the webapp group:
    ```commandline
    poetry install --group webapp
   ```
    For Windows do this also

    ```commandline
    poetry install --group window
   ```

    For Mac do this also
    ```commandline
    poetry install --group mac
   ```

   This will install the package dependencies that are defined in the
`[tool.poetry.group.webapp.dependencies]` section of the `pyproject.toml` file under the
`webapp` group.
4. Verify that the package dependencies are installed correctly by running the following
command:
    ```commandline
    poetry show
    ```

## Activating of Python Virtual Environment

1. Open the terminal / command promp .
2. Navigate to the directory of your project that contains the virtual environment.
3. Run the following command to activate the virtual environment:
    #### On macOS
    ```zsh
    source <venv>/bin/activate
    ```
   #### On windows
    ```zsh
    <venv>\Scripts\activate
    ```
4. Once the virtual environment is activated, you should see the name of the virtual
environment in the command prompt. For example:
    ```commandline
    (venv) C:\Users\username\project>
   ```