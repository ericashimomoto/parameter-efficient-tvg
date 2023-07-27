# Parameter-Efficient Training in Temporal Video Grounding

Code accompanying the paper [Towards Parameter-Efficient Integration of Pre-Trained Language Models In Temporal Video Grounding](https://aclanthology.org/2023.findings-acl.829/). 

This repository includes:

* Code for training and testing the TVG models with adapters 
in the Charades-STA, Activity-Net and YouCookII datasets.
* If you would like to get checkpoints for any of the TVG model and adapter combination we present on the paper, please contact Erica at kidoshimomoto.e@aist.go.jp

# Installation

1. Clone this repo                                                                                                              
   ```bash
   git clone https://github.com/ericashimomoto/parameter-efficient-tvg
   cd parameter-efficient-tvg
   ```

2. Create a conda environment based on our dependencies and activate it

   ```bash
   conda create -n <name> --file environment.yml
   conda activate <name>
   ```

   Where you can replace `<name>` by whatever you want.

2. Download everything
   ```bash
   sh ./download.sh
   ```
   This script will download the following things in the folder `~/data/parameter-efficient-tvg`: 
   * The `glove.840B.300d.txt` pretrained word embeddings.
   * The I3D features for all three datasets, used by all the TVG models.
   * The object features for all three datasets, used by DORi.

   If you would like to change the default output folder for these downloads, please run `sh ./download.sh <download_path>`.

   This script will also install the `en_core_web_md` pre-trained spacy model, and download weights of our model pre-trained on the Charades-STA and Activity-Net datasets on the folders `./checkpoint/chares_sta` and `./checkpoint/anet` respectively.
   
   Downloading everything can take a while depending on your internet connection, please be patient. 

   Note: These links were taken from the original repositories for TMLGA and DORi. If you find any issues with the links, please, contact the respective authors.

## Configuration
 If you have modified the download path from the defaults in the script above please modify the contents of the file `./config/settings.py` accordingly.
  
# Training

To train any of the TVG models, use the experiments recipes specified in the `./experiments` folder:
```bash
python main.py --config-file=experiments/DATASET/EN.yaml
```

Where DATASET is `CharadesSTA`, `ANetCap` or `YouCookII`, and EN is the experiment name. The experiment name is formated as follows: DATASET_TVGMODEL_LM_ADAPTER.yaml

So, by running the following:
```bash
python main.py --config-file=experiments/ANetCap/ANetCap_dori_bert_compacter.yaml
```

You will train DORi on ActivityNet, using BERT as a language model and the Compacter adapter.

These recipes already define the hyper-paramenters used to achieve the results reported in our paper.

We use tensorboardX to visualize progress of our model during training. Please run the followig command to see launch tensorborad:  
```bash
tensorboard --logdir=experiments/visualization/DATASET/EN
```

# Testing

To load pre-trained models and test them, first make sure the weigths have been downloaded and are in the `./checkpoints/EN` foldel. 

Then, edit the respective experiment recipe file by modifiying the variable `ENGINE_STATE` to `TESTER` and include:

`TEST.MODEL: path/to/weights`

Finally, just run:
```bash
python main.py --config-file=experiments/EN.yaml
```


# Citation

If you use our code or data please consider citing our work and the respective TVG models and adapters papers.

```bibtex
@inproceedings{kido-shimomoto-etal-2023-towards,
    title = "Towards Parameter-Efficient Integration of Pre-Trained Language Models In Temporal Video Grounding",
    author = "Kido Shimomoto, Erica  and
      Marrese-Taylor, Edison  and
      Takamura, Hiroya  and
      Kobayashi, Ichiro  and
      Nakayama, Hideki  and
      Miyao, Yusuke",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.829",
    pages = "13101--13123",
}
```
