# sentence-VAE
This is a PyTorch implementation of variational auto-encoder(VAE) for natural languages.

## Usage instructions
### Train model
```
python train.py \
--train_file <file_path> \
--valid_file <file_path> \
--vocab_file <file_path>
```
- ```train_file``` one-sentence-per-line raw corpus file for training.
- ```valid_file``` one-sentence-per-line raw corpus file for validation.
- ```vocab_file``` vocabulary file.
