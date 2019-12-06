import os
import nltk
import rouge
import argparse


def main(args):
    hypotheses = []
    references = []

    with open(os.path.join(args.data_dir, 'hypothesis.txt'), 'r') as f:
        for line in f:
            hypotheses.append(line.strip())
    with open(os.path.join(args.data_dir, 'reference.txt'), 'r') as f:
        for line in f:
            references.append(line.strip())

    evaluate_bleu(hypotheses, references)
    evaluate_rouge(hypotheses, references)


def evaluate_bleu(hypotheses, references):
    print('BLEU')
    score = 0
    for ref, hyp in zip(references, hypotheses):
        ref = ref.lower().split()
        hyp = hyp.lower().split()
        score += nltk.translate.bleu_score.sentence_bleu([ref], hyp, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

    score /= len(hypotheses)

    print("[BLEU]\tScore: %2.2f" % (100 * score))
    print()


def evaluate_rouge(hypotheses, references, max_n=4):
    print('ROUGE-N')
    evaluator = rouge.Rouge(
        metrics=['rouge-n'],
        max_n=max_n,
        limit_length=True,
        length_limit=100,
        length_limit_type='words',
        apply_avg=True,
        apply_best=False,
        alpha=0.5, # Default F1_score
        weight_factor=1.0,
        stemming=True)

    rg = {"rouge-%d" % n: {'f': 0, 'r': 0, 'p': 0} for n in range(1, max_n+1)}
    for hyp, ref in zip(hypotheses, references):
        score = evaluator.get_scores(hyp, ref)
        for n in range(1, max_n+1):
            rg["rouge-%d" % n]['f'] += score["rouge-%d" % n]['f']
            rg["rouge-%d" % n]['r'] += score["rouge-%d" % n]['r']
            rg["rouge-%d" % n]['p'] += score["rouge-%d" % n]['p']

    for n in range(1, max_n+1):
        rg["rouge-%d" % n]['f'] /= len(hypotheses)
        rg["rouge-%d" % n]['r'] /= len(hypotheses)
        rg["rouge-%d" % n]['p'] /= len(hypotheses)

    for n in range(1, max_n+1):
        print("[ROUGE-%d]\tF1: %2.2f\tRecall: %2.2f\tPrecison: %2.2f" \
            % (n, 100 * rg["rouge-%d" % n]['f'], 100 * rg["rouge-%d" % n]['r'], 100 * rg["rouge-%d" % n]['p']))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='result')

    args = parser.parse_args()

    main(args)