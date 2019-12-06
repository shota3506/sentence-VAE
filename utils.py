def decode_sentnece_from_token(tokens, i2w, eos_idx):
    sentence = []
    for t in tokens:
        if t == eos_idx:
            break
        sentence.append(i2w[t])
    return " ".join(sentence)
