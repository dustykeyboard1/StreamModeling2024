import os

def main():

    ### We are starting in the StreamModelling2024 directory, so these two lines
    ### Give us the output locations of our files

    matlab = os.path.join("MATLAB", "HFLUX outputs")
    python = os.path.join("Python", "Results", "CSVs")
    output_location = os.path.join(os.getcwd(), "Python", "Results", "CSVs")
    output = open(os.path.join(output_location, "csv_diff.txt"), "w")  # file to write to

    for mat in os.listdir(matlab):
        for py in os.listdir(python):
            if (os.path.splitext(mat)[1] == ".csv" and
                os.path.splitext(py)[1] == ".csv" and 
                mat == py): # only compare them if they're csv files and have the same name
                    compare_files(os.path.join(matlab, mat), os.path.join(python, py), output, py)

    output.close()

def compare_files(mat, py, output, filename):
    comparison_value = 1E-10    # acceptable difference for matlab vs. python output

    ### Seperating everything by commas to get a list where each item is a csv element
    mat = [item for items in open(mat).readlines() for item in items.split(",")]
    py = [item for items in open(py).readlines() for item in items.split(",")]
    output.write("Filename: " + filename + "\n")
    for i in range(min(len(mat), len(py))):
        mat_val = float(mat[i])
        py_val = float(py[i])
        if (abs(mat_val - py_val) > comparison_value):
            output.write("Index: " + str(i) + 
                         " Matlab val=" + str(mat_val) + 
                         " Python val=" + str(py_val) + 
                         " Difference=" + str(abs(mat_val - py_val)) + "\n")
    
    if (len(mat) != len(py)):
        if (len(mat) > len(py)):
            print("matlab output has " + str(len(mat) - len(py)) + " more entries than python output")
        else:
            print("python output has " + str(len(mat) - len(py)) + " more entries than matlab output")

if __name__=="__main__":
     main()