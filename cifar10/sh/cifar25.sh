gpu=$1
python -u main.py --gpu ${gpu} --epochs 400 --data data/cifar.25.npz --nn_num 32 --loc data/cifar.25_loc.npz --sparse data/cifar25_sparse.npz --adjacency 2d --model DSGC --seed 1234 
