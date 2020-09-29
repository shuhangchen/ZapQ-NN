import argparse


def GetParameters():

    parser = argparse.ArgumentParser(description='ZapNN')

    parser.add_argument('--model-dir', metavar='DIR', help='path to model', default='./model_saved/')
    parser.add_argument('--result-dir', metavar='DIR', help='path to results', default='./result_saved/')

    parser.add_argument('--episodes', default=20000, type=int, help='maximum episode')

    args = parser.parse_args()

    return args
