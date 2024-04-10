import os

min_rows = 7
max_rows = 11
min_cols = 7
max_cols = 11
base_dir = "data/trained/som/size_exp/"

# Create folders for each combination of rows and columns
for rows in range(min_rows, max_rows + 1):
    for cols in range(min_cols, max_cols + 1):
        folder_name = f"SOM_{rows}x{cols}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        # Create 10 .txt files in each folder
        for i in range(10):
            file_name = f"{i}.txt"
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "w") as f:
                f.write("This is a sample text.")

print("Folders and files created successfully.")
