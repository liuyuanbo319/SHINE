import argparse

from experiment import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHINE')
    parser.add_argument('--dataset', type=str, default='ECG200', help='dataset to be trained')
    parser.add_argument('--model_name', type=str, default='SHINE', help='the model to be trained')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of consecutive run with different seeds')
    parser.add_argument('--run_name', default='FCN', type=str,
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')

    parser.add_argument('--seed', default=2024, type=int, help='random seed')
    parser.add_argument('--z_dim', default=12, type=int, help='dimension of the representation z')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or gpu')
    parser.add_argument('--iters', type=int, default=5000, help='The number of iterations')
    parser.add_argument('--show_step', type=int, default=100, help='The number of iterations to print loss')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--result_path', default='.\\results', type=str, help='The folder name used to save results')
    parser.add_argument('--alpha', default=0.1, type=float, help='the smooth coefficient')
    parser.add_argument('--aug_type', default='ws', type=str, help='the augmentation type for SHINE. s: strong. w: weak. o: original')
    parser.add_argument('--K', default=256, type=int, help='the number of sin base')
    parser.add_argument('--P', default=4, type=int, help='the number of trend base')
    parser.add_argument('--preprocessed_dir', default='F:\TS_DATA\dataset\ECG\\ECG200', type=str, help='the preprocessed data dir')

    args = parser.parse_args()
    print(args)
    # analyze_model_efficiency(args)
    train(args)

