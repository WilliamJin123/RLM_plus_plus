from pathlib import Path
from datasets import load_dataset

# Load just the metaphors file you created
DIR = Path(__file__).resolve().parent.as_posix()

# Or load both at once
data_files = {
    "metaphors": DIR + "/filtered_oolong_parquet/metaphors_1024000_plus.parquet", 
    "negation": DIR + "/filtered_oolong_parquet/negation_1024000_plus.parquet"
}
# ds_combined = load_dataset("parquet", data_files=data_files)
# print(ds_combined)
if __name__=="__main__":
    ds_metaphors = load_dataset("parquet", data_files=data_files["metaphors"])["train"]
    print(len(ds_metaphors), "examples in metaphors dataset")