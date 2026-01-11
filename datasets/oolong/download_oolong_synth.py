from datasets import load_dataset
import pandas as pd
# Downloads to /path/to/your/folder
ds = load_dataset("oolongbench/oolong-synth", cache_dir=".")
print("Type of object:", type(ds))
print("Keys/Splits available:", list(ds.keys()))
print("Column names:", ds["test"].column_names)

unique_values = ds["test"].unique("dataset") 
print("Unique values:", unique_values)

df = ds["test"].to_pandas()

for dataset_name in unique_values:
    df_subset = df[df["dataset"] == dataset_name]
    print(f"Dataset: {dataset_name}, Number of examples: {len(df_subset)}")

    print(f"Counts for '{dataset_name}':")
    cutoff = 8000

    while True:
        count = len(df_subset[df_subset["context_len"] >= cutoff])
        print(f"Context >= {cutoff:<10}: {count} examples")
        
        if count == 0:
            break
        
        if cutoff < 512000:
            cutoff *= 2
        else:
            cutoff += 512000

