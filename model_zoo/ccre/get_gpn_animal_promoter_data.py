import h5py
import numpy as np
from tqdm import tqdm

from datasets import load_dataset

ds = load_dataset("songlab/gpn-animal-promoter-dataset")


# Create mapping function with error handling
def encode_sequence(seq):
    mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}
    encoded = []
    for nucleotide in seq:
        if nucleotide in mapping:
            encoded.append(mapping[nucleotide])
        else:
            # Handle unknown characters (N, ambiguous bases, etc.)
            print(f"Unknown character: '{nucleotide}' - skipping or replacing with 4")
            encoded.append(4)  # Use 4 for unknown/ambiguous bases
    return np.array(encoded, dtype=np.uint8)

# Save to HDF5
with h5py.File('dataset.h5', 'w') as f:
    for split_name, dataset in ds.items():
        print(f"Processing {split_name} split...")
        group = f.create_group(split_name)

        # Save IDs as bytes (convert strings to bytes)
        print("Saving IDs...")
        id_data = [str(id_val).encode('utf-8') for id_val in tqdm(dataset['id'], desc="Encoding IDs")]
        group.create_dataset('id', data=id_data)

        # Encode and save sequences
        print("Encoding sequences...")
        encoded_seqs = []
        for seq in tqdm(dataset['seq'], desc="Encoding sequences"):
            encoded = encode_sequence(seq)
            encoded_seqs.append(encoded)

        # For variable length sequences, use special dtype
        print("Saving sequences to HDF5...")
        dt = h5py.special_dtype(vlen=np.uint8)
        seq_dataset = group.create_dataset('seq', (len(encoded_seqs),), dtype=dt)
        
        # Write sequences one by one
        for i, seq in enumerate(tqdm(encoded_seqs, desc="Writing sequences")):
            seq_dataset[i] = seq