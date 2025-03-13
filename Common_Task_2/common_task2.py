# %%
import pandas as pd
import numpy as np
import os

df = pd.read_csv("../Feynman_with_units/I.6.2", 
                 sep=r"\s+",  # whitespace separator
                 header=None).head(1000) # no header row
print(df.shape)

# %%
df.head()

# %%
def batch_encode_p1000(numbers):
    """
    Encode a batch of numbers using the P1000 scheme.
    
    Parameters:
      numbers (np.ndarray): Array of numbers to encode.
      
    Returns:
      sign_tokens (np.ndarray): Array of sign tokens ('+' or '-').
      mantissa_tokens (np.ndarray): Array of mantissa tokens as zero-padded strings.
      exponent_tokens (np.ndarray): Array of exponent tokens as strings (e.g., 'E+3').
    """
    # Convert input to numpy array
    numbers = np.array(numbers, dtype=float)  # Ensure floating-point
    
    # Sign tokens: vectorized assignment based on the sign of the numbers
    sign_tokens = np.where(numbers < 0, '-', '+')
    
    # Work with absolute values
    abs_numbers = np.abs(numbers)
    
    # Handle zeros separately to avoid log10 issues
    non_zero = abs_numbers > 0
    exponent = np.zeros_like(abs_numbers, dtype=int)
    normalized = np.zeros_like(abs_numbers, dtype=float)
    
    # Compute exponent where numbers are non-zero
    exponent[non_zero] = np.floor(np.log10(abs_numbers[non_zero])).astype(int)
    
    # Compute normalized values for non-zero numbers - using float powers to avoid integer power error
    normalized[non_zero] = abs_numbers[non_zero] / np.power(10.0, exponent[non_zero])
    
    # Scale the normalized values to get mantissa in [0, 1000)
    # Multiply by 100 since normalized values are in [1, 10)
    mantissas = np.round(normalized * 100).astype(int)
    
    # Correct any mantissa that rounds to 1000
    overflow = mantissas >= 1000
    if np.any(overflow):
        mantissas[overflow] = 100
        exponent[overflow] += 1
    
    # Format mantissas as 3-digit strings
    mantissa_tokens = np.array([f"{m:03d}" for m in mantissas])
    
    # Format exponent tokens as strings (e.g., 'E+3' or 'E-2')
    exponent_tokens = np.array([f"E{'+' if e >= 0 else ''}{e}" for e in exponent])
    
    return sign_tokens, mantissa_tokens, exponent_tokens




    
    


# %%


# Assuming batch_encode_p1000 function is already defined above

def process_file_with_columns(file_path, col_sep_token="[COL_SEP]"):
    """
    Process a single CSV file containing numerical data.
    Encodes each number using the P1000 scheme and preserves column structure by
    inserting a delimiter token between columns.
    
    Returns a list of tokens for the entire file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path,sep=r"\s+",header=None).head(1000)
    
    # We'll build a token list row by row
    token_list = []
    
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        row_tokens = []
        # Process each column in the row separately
        for col in df.columns:
            # Encode the number in this cell; each cell may be a single number.
            # If a cell contains multiple numbers, you might need to adjust this.
            number = row[col]
            # For a single number, get the tokens as a list
            sign_tokens, mantissa_tokens, exponent_tokens = batch_encode_p1000([number])
            # Combine the triplet (they are arrays with one element each)
            number_tokens = [sign_tokens[0], mantissa_tokens[0], exponent_tokens[0]]
            row_tokens.extend(number_tokens)
            # Insert column delimiter after each column (except last column)
            if col != df.columns[-1]:
                row_tokens.append(col_sep_token)
        # Optionally, you can add a row delimiter (e.g., [ROW_SEP]) if needed.
        token_list.extend(row_tokens)
        token_list.append("[ROW_SEP]")
    
    return token_list

# # Example usage for a single file:
# file_path = "../Feynman_with_units/I.6.2"  # Update the path as needed
# token_list = process_file_with_colum                                                                                                                   ns(file_path)
# print(token_list[:20])  # Print first 20 tokens for inspection
# print("Total tokens in file:", len(token_list))


# %%
folder = "../Feynman_with_units"

file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print(file_list)

# %%
print(len(file_list))

# %%
def process_all_files_with_columns(file_list,folder_path, col_sep_token="[COL_SEP]"):
    all_file_tokens = {}
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        tokens = process_file_with_columns(file_path, col_sep_token)
        all_file_tokens[file_name] = tokens
        print(f"Processed {file_name}, token sequence length: {len(tokens)}")
    return all_file_tokens

# Example usage:
folder_path = "../Feynman_with_units"
all_tokens = process_all_files_with_columns(file_list,folder_path)


# %%
from tokenizers import Tokenizer

tokenizer=Tokenizer.from_file("my_tokenizer.json")

token_string=" ".join(token_list)

encoded=tokenizer.encode(token_string)

encoded_lines=[]
encoded_lines.append(encoded)
print(len(encoded_lines[0]))


# %%
from tokenizers import Tokenizer

tokenizer=Tokenizer.from_file("my_tokenizer.json")
tokenized_dict={}
for key,value in all_tokens.items():
    current_token_string=" ".join(value)
    encoding_result = tokenizer.encode(current_token_string)  
    tokenized_dict[key] = encoding_result.ids


    

# %%
import json
with open("dataset_encoded.json", "w") as f:
    json.dump(tokenized_dict, f, indent=4)

# %%



