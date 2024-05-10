import os

def folders_for_diff_sizes(base_dir, min_rows, max_rows, min_cols, max_cols):
    # Create folders for each combination of rows and columns
    for rows in range(min_rows, max_rows + 1):
        for cols in range(min_cols, max_cols + 1):
            for k in range(5):
                folder_name = f"SOM_{rows}x{cols}x{k}"
                folder_path = os.path.join(base_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

                # Create 10 .txt files in each folder
                for i in range(10):
                    file_name = f"{i}.txt"
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "w") as f:
                        f.write("This is a sample text.")

    print("Folders and files created successfully.")


def folders_for_diff_sizes_and_lr(base_dir, learning_rates, min_rows, max_rows, min_cols, max_cols):
    # Create folders for each combination of rows, columns, and learning rates
    for rows in range(min_rows, max_rows + 1):
        for cols in range(min_cols, max_cols + 1):
            for lr in learning_rates:
                folder_name = f"SOM_{rows}x{cols}_lr{lr}"  # Include learning rate in folder name
                folder_path = os.path.join(base_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

                # Create 10 .txt files in each folder
                for i in range(10):
                    file_name = f"{i}.txt"
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "w") as f:
                        f.write("This is a sample text.")

    print("Folders and files created successfully.")

def folders_for_diff_lr(base_dir, alpha_s_values, alpha_f_values, rows, cols):
    
    for lr_s in alpha_s_values:
        for lr_f in alpha_f_values:
            for i in range(5):    
                folder_name = f"SOM_{rows}x{cols}_lr{lr_s}_lr{lr_f}_{i}"  # Include learning rate in folder name
                folder_path = os.path.join(base_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

                # Create 10 .txt files in each folder
                for i in range(10):
                    file_name = f"{i}.txt"
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "w") as f:
                        f.write("This is a sample text.")

    print("Folders and files created successfully.")


min_rows = 8
max_rows = 10
min_cols = 8
max_cols = 10
base_dir = "data/trained/som/lr_exp/"
# Define learning rates to experiment with
alpha_s_values = [0.1, 0.5, 1.0]
alpha_f_values = [0.001, 0.01, 0.1]

folders_for_diff_lr(base_dir, alpha_s_values, alpha_f_values, 10, 10)
#folders_for_diff_sizes(base_dir, min_rows,max_rows,min_cols,max_cols)