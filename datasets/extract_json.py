
import json
import os
import argparse

# =========================================================================
# SPLIT CONFIGURATION
# =========================================================================
# Define the target files and the rules for what goes into them.
# 
# Keys:
#   "filename": The path/name of the output file.
#   "rules":    A dictionary of attribute:values required to get into this file.
#
# Logic: An object is added to a file if it matches ALL rules in that dictionary.
#        An object can be added to multiple files if it matches multiple rules.
# =========================================================================

SPLIT_DEFINITIONS = [
    {
        "filename": "longbenchv2/history_qa.json",
        "rules": {
            "domain": ["Long-dialogue History Understanding"]
            # Add "sub_domain": ["..."] here if you want to be more specific
        }
    },
    {
        "filename": "longbenchv2/code_qa.json",
        "rules": {
            "domain": ["Code Repository Understanding"]
        }
    }
]

# =========================================================================

def matches_rules(obj, rules):
    """
    Returns True if the object matches all key-value pairs in the rules dict.
    """
    for attribute, allowed_values in rules.items():
        # Check if object has attribute and if its value is in our allowed list
        if obj.get(attribute) not in allowed_values:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Split a single JSON file into multiple files based on filters.")
    parser.add_argument("-i", "--input", default="longbenchv2should/data.json", help="Path to input JSON file.")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading input: {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Error: Input JSON root must be a list.")
            return
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.input}'")
        return

    # 2. Prepare Buckets
    # We create a dictionary to hold the lists of objects for each file
    output_buckets = {defn["filename"]: [] for defn in SPLIT_DEFINITIONS}

    # 3. Sort Objects into Buckets
    for obj in data:
        # Check this object against every definition in our list
        for defn in SPLIT_DEFINITIONS:
            filename = defn["filename"]
            rules = defn["rules"]
            
            if matches_rules(obj, rules):
                output_buckets[filename].append(obj)

    # 4. Write Files
    for filename, content in output_buckets.items():
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
            print(f"Saved {len(content)} objects to {filename}")
        except Exception as e:
            print(f"Error writing to {filename}: {e}")

if __name__ == "__main__":
    main()