from datasets import load_dataset, Dataset
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_dna_dataset(dataset, max_length=1024):
    """
    Preprocess the DNA dataset to ensure it's in the correct format for the model.
    Chunks sequences into segments of maximum length 1024 to match GPT-2's context window.
    """
    logger.info(f"Preprocessing DNA dataset with max chunk length of {max_length}...")
    
    # Function to clean DNA Sequences (remove any non-ACGT characters if they exist)
    def clean_sequence(seq):
        if not isinstance(seq, str):
            logger.warning(f"Non-string sequence found: {type(seq)}. Converting to string.")
            seq = str(seq) if seq is not None else ""
        return ''.join(c for c in seq if c in 'ACGT')
    
    # Function to chunk a sequence into parts of max_length
    def chunk_sequence(seq, max_length):
        return [seq[i:i+max_length] for i in range(0, len(seq), max_length)]
    
    # Process and chunk the dataset
    chunked_texts = []
    total_original_sequences = 0
    total_chunked_sequences = 0
    
    dlen = len(dataset)
    max_len = int(0.8 * dlen)

    # Process each example
    for example in dataset:
        total_original_sequences += 1
        if total_original_sequences > max_len:
            break
        
        # Get the sequence (check both possible field names)
        seq = example.get('sequence', example.get('Seq', ''))
        
        # Clean the sequence
        cleaned_seq = clean_sequence(seq)
        
        if not cleaned_seq:
            logger.warning("Empty sequence found after cleaning, skipping")
            continue
            
        # Chunk the sequence
        chunks = chunk_sequence(cleaned_seq, max_length)
        
        # Add each chunk as a separate example
        for chunk in chunks:
            if len(chunk) >= 200:  # Skip very short chunks (less than ~20% of max length)
                chunked_texts.append({"text": chunk})
                total_chunked_sequences += 1
    
    logger.info(f"Original dataset had {total_original_sequences} sequences")
    logger.info(f"Chunked dataset has {total_chunked_sequences} sequences")
    if total_original_sequences > 0:
        logger.info(f"Average chunks per sequence: {total_chunked_sequences/total_original_sequences:.2f}")
    
    # Create a new dataset with the chunked texts
    chunked_dataset = Dataset.from_list(chunked_texts)
    
    if len(chunked_dataset) > 0:
        logger.info(f"Sample chunked text: {chunked_dataset[0]['text'][:100]}...")
    
    return chunked_dataset

def save_to_text(dataset, filepath):
    """
    Save a dataset to a text file (one sequence per line).
    """
    logger.info(f"Saving dataset with {len(dataset)} examples to {filepath}")
    
    with open(filepath, 'w') as f:
        for example in dataset:
            f.write(example['text'] + '\n')
    
    logger.info(f"Dataset saved to {filepath}")

def main():
    logger.info("Loading dataset: simecek/Human_DNA_v0")
    try:
        dataset = load_dataset("simecek/Human_DNA_v0")
        logger.info(f"Dataset loaded successfully with splits: {dataset.keys()}")
        
        # Create output directory
        os.makedirs("processed_dna_data", exist_ok=True)
        
        # Process each split with chunking
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = preprocess_dna_dataset(dataset[split], max_length=1024)
        
        # If there's no validation split, create one from the train split
        if 'validation' not in processed_dataset and 'train' in processed_dataset:
            logger.info("Creating validation split from training data")
            train_val = processed_dataset['train'].train_test_split(test_size=0.1)
            processed_dataset['train'] = train_val['train']
            processed_dataset['validation'] = train_val['test']
        
        # Save the processed dataset as text files
        for split in processed_dataset:
            text_path = f"processed_dna_data/{split}.txt"
            save_to_text(processed_dataset[split], text_path)
        
        logger.info("Dataset processing completed and saved to text files in 'processed_dna_data/'")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    main()