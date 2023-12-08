# 2024 CS 410 Senior Seminar

## Stream Temperature Modeling

### Authors 📝
- Michael Scoleri
- Violet Shi
- James Gallagher

### Customer
Emily Baker, Hamilton College

## ✨ OVERVIEW ✨
The customer had a program written in MATLAB, which is used to model stream temperatures across the country. The MATLAB user must have an excel sheet formatted and a license to use MATLAB. The customer wished for it to be converted over into Python so her product to be open sourced and free to use. There are 3 version of our program in this github:
- Alpha: The basic implementation of the MATLAB code. Extremely slow
- Beta: Major changes from Alpha. Classes were created, multi threading was introduced and matrix multiplication was used. Extremely large time improvement from Alpha
- Version 1: The most recent and up-to-date code base. This includes a GUI and Command Line function for executing the program.

# 💥 VERSION 1 FEATURES 💥
- A fully functioning GUI for MacOS and Windows
- Command Line features for easy execution
- Multi Threading to reduce runtime
- Sensitivity customization
-  Matrix Multiplication to reduce runtime

# ✅ REQUIREMENTS ✅:
- Python3.11: Can be downloaded from this link - https://www.python.org/downloads/
- Pip3: Can be downloaded from this link - https://www.activestate.com/resources/quick-reads/how-to-install-and-use-pip3/
- We reccomend you download github command lines - https://github.com/git-guides/install-git
- We reccomend you use Visual Studio Code with this code - https://code.visualstudio.com/download

# 🖇️ INSTALLATION STEPS 🖇️
- Open your terminal on your operating system
- Clone the repository into you're desired location.
- Change directory into the Python Folder

# 🧪DATA PREPARATION🧪: 
As of right now, the program only accepts data in the form of Excel files, containing multiple sheets. 
- For how the data should be organized, please view the example data in: `StreamModeling2024/Python/Data/example_data/example_data.xlsx`
- All data must be kept in the `StreamModeling2024/Python/Data` directory. 

# 🔋 HOW TO USE 🔋: 
- The Python code must be executed from the `StreamModeling2024/Python` directory.
- To use the command line, run the `Python/Scripts/commandLine.py` file, and follow the on screen instructions.
- To use the GUI feature, run the `Python/Scripts/gui_demo.py` file, and follow the on screen instructions.
- To use the back end execution, run the `Python/Scripts/script_to_run.py` file.
  - Navigate to line 33 and edit the .xlsx file name to your data.
  - Execute the file. 
