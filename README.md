# sentence-VAE
This is a PyTorch implementation of variational auto-encoder (VAE) for natural languages.

## Usage instructions
### Train model
```
python main.py train \
--train_file <file_path> \
--valid_file <file_path> \
--vocab_file <file_path>
```
- ```train_file``` one-sentence-per-line raw corpus file for training.
- ```valid_file``` one-sentence-per-line raw corpus file for validation.
- ```vocab_file``` vocabulary file.

### Sample sentences from prior distribution
```
python sample.py \
--vocab_file <file_path> \
--checkpoint_file <file_path> \
--sample_size 10
```
- ```vocab_file``` vocabulary file.
- ```checkpoint_file``` PyTorch model parameter file.
- ```sample_size``` number of samples to generate.

## Example
### Penn Tree Bank
Download data from [here](https://drive.google.com/drive/folders/1HyeGxhgtWWtTaCYlLOAIlAIXsGXM7TKG?usp=sharing). 
Download trained model parameters and vocabulary file from [here](https://drive.google.com/drive/folders/1NMoPoVttRXJ74zN9W2HMqtSfD_S6OoK8?usp=sharing).

Sentence samples.
```
- a spokesman for the <unk> said he is n't recommending
- to make the comparable market directly comparable each index is based on the close of N equaling N
- you do n't have a <unk>
- but they did n't have to get a <unk> plan
- but pfizer said its third-quarter earnings were hurt by the year earlier
- he said the board had n't yet been scheduled to meet with the situation
- yesterday 's edition of the company 's stock exchange composite trading yesterday
- campeau 's chairman stephen m. wolf had been approached by the end of the year
- these include the tax rate of N N of the common stock and the other N N of the common stock
- but he said the <unk> had been <unk> in the past two years
```
