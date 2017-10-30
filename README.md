#DSGC (Depthwise Separable Graph Convoluiton)

This repo contains the experiment codes for the three experiments in the paper. Because the limitation of the space of Github. I include the data file in the website. 

### File Structure 

In each folder, the subfolder 'models/' contains the codes for all models mentioned in the papaer. 'sh/' contains several example scripts to run the program. 

### Hyper-Parameter Setting

The model structure and hyper-parameters are encoded in the model files. Running the program with the default hyper-parameter, you should get the similar performance as the results in the paper. 

For CIFAR dataset, we encode two different hyper-parameter setting for the origin CIFAR dataset and the subsampled dataset. You can manually switch them. 

Notice that, the program arguments '--adjacency' indicates what is the spatial information to the models. If use the graph convolution approaches, including GCN, ChebyNet and DCNN, you should set '--adjacency Laplacian'. And if use the geometric convolution and our method (DSGC), you should set '--adjacency 2d'. 


