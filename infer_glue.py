from fairseq.models.roberta import RobertaModel
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr
import sys
import os
import argparse


parser = argparse.ArgumentParser(
    description='Inference on GLUE')
parser.add_argument('--task', type=str, default='', help='specify the inference task')
parser.add_argument('--ckpt', type=str, default='', help='specify the path to the checkpoint')
args = parser.parse_args()


task_list = ['CoLA', 'MNLI', 'QNLI', 'RTE', 'STS-B', 'MRPC', 'QQP', 'SST-2', 'WNLI']

assert args.task in task_list
assert os.path.exists(args.ckpt)


data_dir = '/data1/dataset_asr/glue_data/'

roberta = RobertaModel.from_pretrained(
    args.ckpt,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=args.task+'-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)


if args.task == 'RTE':
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(data_dir + args.task + '/dev.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            nsamples += 1
    print('| Accuracy: ', float(ncorrect)/float(nsamples))


elif args.task == 'CoLA':
    ncorrect, nsamples = 0, 0
    predictions = []
    ground_truth = []
    roberta.cuda()
    roberta.eval()
    with open(data_dir + args.task + '/dev.tsv', encoding='utf-8') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent, target = tokens[3], tokens[1]
            tokens = roberta.encode(sent)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            prediction_label = int(prediction_label)
            target = int(target)
            predictions.append(prediction_label)
            ground_truth.append(target)
            nsamples += 1

    print('| Accuracy: ', float(ncorrect)/float(nsamples))
    MCC = matthews_corrcoef(ground_truth, predictions)
    print('| MCC: ', MCC)


elif args.task == 'STS-B':
    gold, pred = [], []
    with open(data_dir + args.task + '/dev.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
            tokens = roberta.encode(sent1, sent2)
            features = roberta.extract_features(tokens)
            predictions = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features)
            gold.append(target)
            pred.append(predictions.item())

    print('| Pearson: ', pearsonr(gold, pred))

else:
    print('Not support yet:', args.task)
    sys.exit()