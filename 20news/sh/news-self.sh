gpu=$1
python -u main.py --gpu ${gpu} --epochs 400 --data data/20news1000_new.npz --nn_num 32 --loc data/embed1000_self.npz --sparse data/20news_sparse_1000_self.npz --adjacency 2d --model DSGC --seed 1234 --embed 0 
