
# GlanceNets: Interpretabile, Leak-proof Concept-based Models
Code implementiation for the homonymous paper. 

This repo was created upon orginal work on **disentanglement-pytorch**, please consider citing also the original library (https://github.com/amir-abdi/disentanglement-pytorch).
   
The following VAE variants can be included to the original GlanceNet loss:
- VAE
- β-VAE ([Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf))
- Info-VAE ([InfoVAE: Information Maximizing Variational Autoencoders](https://arxiv.org/abs/1706.02262))
- Beta-TCVAE ([Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942))
- DIP-VAE I & II ([Variational Inference of Disentangled Latent Concepts from Unlabeled Observations ](https://openreview.net/forum?id=H1kG7GZAW))

***Note:*** *Everything* is modular, you can mix and match neural architectures and algorithms.
Also, multiple loss terms can be included in the `--loss_terms` argument, each with their respective weights. This enables us to combine a set of disentanglement algorithms for representation learning. 


### Requirements and Installation

Install the requirements: `pip install -r requirements.txt` \
Or build conda environment: `conda env create -f environment.yml`

The library visualizes the ***reconstructed images*** and the ***traversed latent spaces*** and saves them as static frames as well as animated GIFs. It also extensively uses the web-based Weights & Biases toolkit for logging and visualization purposes.

***Note:** The leakage test on MNIST is implemented with a newer version of pytorch, which includes tensorboard loggers. Be sure to create a new environment with `conda env create -f MNIST_TEST/mnist_env.yml` or by installing the dependencies with `pip install -r MNIST_TEST/mnist_requirements.txt`. Using the original environment may cause conflicts. * 
### Dataset setup

We considered variants of celebA, dSprites and MNIST to perform our experiments. In particular, the datasets dsprites_leakage, mpi3dtoy and celebA-64 must be preprocessed in order to be used during pipelines. Please consider to place all of them to the same folder and set `$DISENTANGLEMENT_LIB_DATA` environment variable to the directory holding all the datasets , or choose everytime with the flag `--dset_dir`   (the latter is given priority). \ 

Different datasets are called with `--dset_name` flag or or the `$DATASET_NAME` environment variable to the name of the dataset (the former is given priority).  The supported datasets are: 
[mnist](http://yann.lecun.com/exdb/mnist/),
[dsprites](https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz),
[mpi3d_toy](https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz),
[celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

**Pre-processing:** 
Different preparations are included in the compressed file `preprocessing.zip` but are not automatized. Be sure to have at least 20 Gb  for datasets storage. Extract the content of `preprocessing.zip` to the selected data folder (default: `data/`). For each dataset, some preprocessing is required. In order:
- **dSprites:**  download the original dataset from https://github.com/deepmind/dsprites-dataset and make sure to move the .npz file to the chosen datafolder; 
- **dSprites_leakage:** run the script `prepare_leakage_dsprites.py` to create the train and test .npz files for concept leakage; 
- **MPI3D:**  download the original .npz folder from https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz for toy version.  Then, run the script `mpi_prepare_labels.py` to update the npz archive with latent factors;
- **CelebA:** download the original dataset from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset and extract the content in a selected folder.  Move the folder `preprocess_celebA/` to the same folder and run the script `preprocess_celebA-64.sh` to rescale all images to 64x64 resolution. Then, run the python file `create_labels.py` to include only the selected attributes to the dataset. At the end of the process there will be a npz archive containg fewer data;
- **MNIST:** The download procedure is  automatized in the code. Be sure to have enough free space in the original project folder.

**Classes selection:**
Each dataset comes with its labels, but they are generated in different ways. Labels are predefined in dSprites, but they can be changed following the paper procedure. For MPI3D and CelebA, be sure to include the `km.pkl` files into the correct folder and with the correct name. In `preprocessing.zip`  they are collected in the respective folders (the cluster creation is also included in those folders). Labels for MNIST and dSprites_leakage are automatically included in the original datasets. 
 

### Training

Training processes can be either launched alone with:

    python main.py [[--ARG ARG_VALUE] ...]

or to reproduce experiments:

    bash scripts/SCRIPT_NAME
    
  
    
#### 1) Important Flags
- `--alg`: The model for training.  ***Values**:  GrayVAE_Join, CBM_Join* 
- `--masking_fact`: Selects the percentage of supervised training examples possessing latent factors information. It is a float variable in the interval [0,100]. 
- `--conditional_prior`: Choose between the GlanceNet conditional prior (True) and the unconditional one (False).  The conditional prior requires class labels. ***Values**: True, False.* 

- `--loss_terms`: Extensions to the VAE algorithm  are implemented as plug-ins to the original formulation.  As a result, if the loss terms of two learning algorithms (*e.g.*, A and B)  were found to be compatible, they can simultaneously be included in the objective function with the flag set as:
		 `--loss_terms A B`.  The `loss_terms` flag can be used only with GRAYVAE algorithm. \
   ***Values**:  DIPVAEI, DIPVAEII, BetaTCVAE, INFOVAE, and FACTORVAE (not available for now)*.
    
- `--evaluation_metric`: Metric(s) to use for disentanglement evaluation (see `scripts/aicrowd_challenge`). \
***Values**: mig, sap_score, irs, factor_vae_metric, dci, beta_vae_sklearn*

**WARNING:** disentanglement metrics work only on dSprites and MPI3D, for now. Be sure not to include the evaluation process on other datasets.   

#### 2) Following Training/Validation/Test Performances
In this current library implementation, tensorboard loggers are **not** available. The training/validation/test losses are saved in the `/logs` folder, with the name of the dataset and of the model chosen. It is possible to choose the log directory name setting the variable `--out_path` to the desired string, otherwise a new one will be created with the date time information. Jupyter notebooks are available in the folder `analysis/` but existing implementations can be customized to your liking. 

#### 3) Saving and Loading Checkpoints
The current implementation allows to choose where to save the model checkpoints and whether to load trained models. To access them use the flags:
- `--ckpt_dir` to select the folder where to save model parameters (default: `checkpoints/`):
- `--ckpt_load` to load an existing trained model. Be sure to call it from `your_path/last`.

#### 4) Replicate the experiments
The folder scripts contains various sh configurations to replicate our paper experiments. In particular, GlanceNet and CBNM training pipelines can be found for *dSprites, mpi3D_toy* and *CelebA*. One can vary `--seed` and `--masking_fact`  flags to choose between different scenarios.   Be sure to select the correct environment (default name `dis`). Also, the script `leakage_dsprites.sh` contains the specifications to test leakage on either *GlanceNet, CBNM, and VAE*.

On the other hand, it is sufficient to run `MNIST_TEST/parity_test.py` to replicate *MNIST* leakage test. Be sure to select the correct environment (default name `mnist_test`).


### Evaluate the performances
Evaluation of performances is done post training, from saved .csv and .npy files in logs folders. In particular, the trend of training losses can be found in the  `log_name/train_runs/metrics.csv` and those on test in `log_name/eval_results/test_metrics.csv`. If disentanglement metrics are used during training, evaluations are saved in `log_name/eval_results/dis_metrics.csv`.  The log folder includes also the latent factors and their model encodings, both for training and test, in  `latents_obtained.npy`. Downstream prediction and groundtruth values are saved into `downstream_obtained.npy`.

- **Accuracy:** The model accuracy is evaluated with sklearn.metrics from  `log_name/eval_results/downstream_obtained.npy`;

- **Alignment:** The model alignmnent is evaluated with the aid of `disentanglement_lib` for dSprites and MPI3D, and it is contained in `log_name/eval_results/dis_metrics.csv`. For CelebA, the performances are calclulated from `log_name/eval_results/latents_obtained.npy`;  
- **Explicitness:** Similarly, for CelebA the performances are calclulated from `log_name/eval_results/latents_obtained.npy`. Explicitness for dSprites and MPI3D can be found into the `dci/.../evaluation_result.json` from the saved checkpoint.

Visualization and code implementations are present inside some jupyter notebooks in the folder `analysis/`.


### Contributions
Any contributions, suggestions or questions are welcome. Feel free to submit bugs, feature requests, or questions as issues, or contact me directly via email at:  

