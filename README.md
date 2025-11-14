# Python Project Template <!-- omit from toc -->

A project template for writing Python code on a mac.

- [Downloading Files](#downloading-files)
- [Initializing the project](#initializing-the-project)
- [Running your code](#running-your-code)
- [Using VSCode](#using-vscode)
- [Adding and removing dependencies](#adding-and-removing-dependencies)


## Downloading Files

If you have a github account you can use this repository as a template to
create your own project.  (See the green "Use this template" button toward
the upper-right corner of the page.)

If you do not have a github account, you can download the files directly as follows:

1. Click the green "Code" button at the top of the repository
2. Select "Download ZIP"
3. Extract the ZIP file to your desired location
4. Open Terminal and navigate to the extracted folder:
   ```bash
   cd path/to/extracted-folder
   ```

## Initializing the project

1. **Install uv**

   `uv` is a tool to streamline the setup and management of Python projects.

   ```sh
   # Check if it's already installed
   uv --version
   # Install if needed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Verify installation
   uv --version
   ```

2. **Set up virtual environment**

   A "virtual environment" is a self-contained directory that contains the Python
   version you have chosen, along with the packages required by your project.
   This allows you to manage dependencies for your project without affecting (or
   being affected by) other Python projects on your system.

   The following command creates a virtual environment with a specific python version
   (in this example `3.13`)
   ```sh
   # Install the python version if needed, and set for this project.
   uv python install 3.13
   uv python pin 3.13
   # Create the virtual environment
   uv venv --prompt venv
   # Enter the virtual environment, and install dependencies
   source .venv/bin/activate
   uv sync
   ```

## Running your code

1. **Entering the virtual environment**

   Whenever you want to run your code, make sure you are in the virtual environment.  To enter the virtual environment, run:

   ```bash
   source .venv/bin/activate
   ```
   Note that:
   * `(venv)` will be prepended to your terminal prompt to indicate
   that you are in the virtual environment.
   * You can run `deactivate` to exit the virtual environment (but there's usually no need to do this).
   * There's no harm if you inadvertently re-run the `.source venv/bin/activate` command when you don't need to.


2. **Pythoning the python**

   To run your code, run `python <path-to-your-code>`.  For example:

   ```bash
   python my_project/hello_world.py
   ```

   Note that the file `hello_world.py` is in the `my_project` directory (feel free to rename it), following the usual python convention of keeping your code in a subdirectory of the project.

## Using VSCode

After installing VSCode, just type `code .` in the project root directory.

There are many extensions that are useful for python development; this repository comes with some specific recommendations.  To install them:
* Open the command palette (Ctrl+Shift+P) and type "Extensions: Show Recommended Extensions".
* Wait for the list of recommended extensions to appear, it's a bit slow.
* Click "Install" on the Workspace Recommended extensions.
* Once the installations complete (it should be fast), quit VScode and then type `code .` to reopen the project.

The recommended extensions include linting functionality (to catch errors before you run your code) and auto-formatting.

## Adding and removing dependencies

To add packages to your project, use the `uv add` command:
```bash
# Add a single package ("requests")
uv add numpy
# Recommended: require a minimum version -- in this case, numpy 2.3 or higher
uv add "numpy>=2.3"
```

To remove packages:
```bash
uv remove numpy
```
