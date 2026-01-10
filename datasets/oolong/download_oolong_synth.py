from datasets import load_dataset

# Downloads to /path/to/your/folder
ds = load_dataset("oolongbench/oolong-synth", cache_dir=".")
print("Type of object:", type(ds))
print("Keys/Splits available:", list(ds.keys()))
print("Column names:", ds["test"].column_names)

unique_values = ds["test"].unique("dataset") 
print("Unique values:", unique_values)