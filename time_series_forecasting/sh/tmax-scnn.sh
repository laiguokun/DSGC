gpu=$1
python -u main.py --epochs 200 --normalize 1 --mask --dropout 0. --window 6 --seed 12345 --horizon 1 --gpu ${gpu} --data data/USHCN_TMAX.txt --loc data/USHCN_loc.txt --embed 0 --nn_num 32 --adjacency 2d --model SCNN --num_layers 10 --hidCNN 100
