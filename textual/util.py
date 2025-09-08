import os


# removes ids deemed bad from all txt files from datasets eg. humanml3d/texts
def remove_bad_ids(data_dir, bad_ids):
    files = ["all.txt", "train.txt", "val.txt", "test.txt", "train_val.txt"]

    for fname in files:
        with open(os.path.join(data_dir, fname), "r") as f:
            lines = f.readlines()
        with open(os.path.join(data_dir, fname), "w") as f:
            for line in lines:
                if line.strip() not in bad_ids:
                    f.write(line)
