import os
from datasets import load_dataset

# 1. Load the dataset (using 'test' split as per your previous code)
ds = load_dataset("oolongbench/oolong-synth", split="test", cache_dir=".")

# 2. Configuration
target_categories = ["metaphors", "negation"]
min_context_len = 1_024_000
output_dir = "filtered_oolong_parquet"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for category in target_categories:
    print(f"Processing category: {category}...")
    
    # 3. Filter for the specific category AND the length cutoff
    # We use the dataset object's .filter() method which is efficient
    subset = ds.filter(
        lambda x: x["dataset"] == category and x["context_len"] >= min_context_len
    )
    
    # Check if we actually found data
    if len(subset) == 0:
        print(f"Warning: No examples found for {category} with length >= {min_context_len}")
        continue

    # 4. Define filename
    # Example: filtered_oolong_parquet/metaphors_128k.parquet
    file_path = os.path.join(output_dir, f"{category}_{min_context_len}_plus.parquet")
    
    # 5. Save to Parquet
    # 'compression="zstd"' is highly recommended for disk space efficiency 
    # while remaining fast to read.
    subset.to_parquet(file_path, compression="zstd")
    
    print(f"Saved {len(subset)} examples to: {file_path}")

print("\nAll done!")