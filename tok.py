from transformers import GPT2Tokenizer, GPT2TokenizerFast
import os

# Create a directory for our tokenizer
os.makedirs("dna_tokenizer", exist_ok=True)

# Define the DNA vocabulary
vocab = {
    "<|endoftext|>": 0,  # Special token from GPT-2
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4
}

# Create a merge file (for BPE, but we won't use it much since our tokens are single characters)
with open("dna_tokenizer/merges.txt", "w") as f:
    f.write("#version: 0.2\n")  # Header required by the tokenizer

# Write the vocabulary to a file
with open("dna_tokenizer/vocab.json", "w") as f:
    import json
    json.dump(vocab, f)

# Create the tokenizer using the vocab and merges files
tokenizer = GPT2TokenizerFast(
    vocab_file="dna_tokenizer/vocab.json",
    merges_file="dna_tokenizer/merges.txt",
)

# Test the tokenizer
test_sequence = "ACGTACGTACGT"
tokens = tokenizer.encode(test_sequence)
print(f"Encoded tokens: {tokens}")
decoded = tokenizer.decode(tokens)
print(f"Decoded text: {decoded}")

# Save the tokenizer
tokenizer.save_pretrained("dna_tokenizer")

print("DNA tokenizer created and saved to 'dna_tokenizer' directory.")