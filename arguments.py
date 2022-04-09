import argparse
import json

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--dataset', type=str, default='yahooR3', help='Choose from {yahooR3, coat, simulation}')
    parser.add_argument('--uniform_ratio', type=float, default=0.05, help='the ratio of uniform set in the unbiased dataset.')
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    parser.add_argument('--type', type=str, default='explicit', help='feedback type. implicit,explicit')
    parser.add_argument('--val_diff', type=str, default='None', help='if it can be different, uniform or bias data to val')

    # parser.add_argument('--teacher_model_args', type=json.loads, default='{"emb_dim": 10, "learning_rate": 0.1, "weight_decay": 10}', help='weight model arguments.')
    parser.add_argument('--exp_name', type=str, default=' ',help='to save best model weight with this name')
    parser.add_argument('--gama', type=float, default='0.17',help='CFF gama')
    parser.add_argument('--gama2', type=float, default='1.25',help='CFF_A gama')
    parser.add_argument('--beta', type=float, default='0.01',help='A value for fine-tune loss_A')
    parser.add_argument('--epoch', type=int, default=None,help='A value for fine-tune loss_A')
    return parser.parse_args()
