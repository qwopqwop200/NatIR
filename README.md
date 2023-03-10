# NatIR
NatIR: Image Restoration Using Neighborhood-Attention-Transformer

**This code is based on [Fried Rice Lab](https://github.com/fried-rice-lab/friedricelab) and Neighborhood-Attention(https://github.com/SHI-Labs/NATTEN)**

## Comparison with SOTA model
| Model      |     Dataset    | Params(M) |     Set5     |     Set14    |    BSD100    |   Urban100   |   Manga109   |
| ---------  | -------------- | --------  | ------------ | ------------ | ------------ | ------------ | ------------ |
| [SwinIR](https://arxiv.org/abs/2108.10257)     |      DIV2K     |   0.90     | 32.44/0.8976 | 28.77/0.7858 | 27.69/0.7406 | 26.47/0.7980 | 30.92/0.9151 |
| [ESWT](https://arxiv.org/abs/2301.09869)       |      DIV2K     |   0.58     | 32.46/0.8979 | 28.80/0.7886 | 27.70/0.7410 | 26.56/0.8006 | 30.94/0.9136 |
| [EDT](https://arxiv.org/abs/2112.10175)        | DIV2K+Flickr2K |   0.92     | 32.53/0.8991 | 28.88/0.7882 | 27.76/0.7433 | 26.71/0.8051 | 31.35/0.9180 |
| [GRL-T](https://arxiv.org/abs/2303.00748v1)      |        ?       |   0.91     | 32.56/0.9029 | 28.93/0.7961 | 27.77/0.7523 | 27.15/0.8185 | 31.57/0.9219 |
| **NatIR(ours)**|      DIV2K     |   0.98     | 32.58/0.8995 | 28.90/0.7886 | 27.76/0.7428 | 26.68/0.8035 | 31.25/0.9169 |

Unlike SwinIR, it uses Neighborhood Attention.

Unlike ESWT, Unlike ESWT, it has a small window size of 13. ESWT uses window sizes (24 x 6 and 6 x 24). This gives a receptive field of the same size at a fraction of the cost compared to a 24 x 24 window size. Also, the HR patch size is 288 (I use 128). According to [ESRGAN](https://arxiv.org/abs/1809.00219), this results in additional performance improvements.

Unlike EDT, it does not use the Flickr2K dataset. Because of this, it is inappropriate for a fair comparison. Additional use of Flickr2K has performance benefits.

Unlike GRL, it uses only local attention. Therefore, additional performance improvement may be possible through global attention.
## Next steps
I believe that further performance improvements can be achieved through the following improvements:
1. Like EWST, a 24 x 6 window size can be used. Unfortunately, this is not supported by natten.
2. A wider receptive field can be obtained through [Dilated Neighborhood Attention](https://arxiv.org/abs/2209.15001)(sparse global attention). However, this requires a larger HR patch size.
3. Receptive fields of various sizes can be obtained using [Hydra-NA](https://arxiv.org/pdf/2211.05770.pdf). However, this requires a larger HR patch size.
4. Like GRL, it additionally uses channel attention.
5. Use a more powerful optimizer (e.g [adan](https://arxiv.org/abs/2208.06677), [lion](https://arxiv.org/abs/2302.06675), etc), a more powerful learning rate scheduler (e.g [warm up](https://arxiv.org/abs/1706.02677), [cosine annealing](https://arxiv.org/abs/1608.03983), etc), and a larger HR patch size.
6. use the Flickr2K dataset
## How to Use

### 1 Preparation

#### 1.1 Environment

Use the following command to build the Python environment:

```shell
conda create -n frl python
conda activate frl
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # Mainland China only!
pip install torch torchvision basicsr einops timm matplotlib
pip install natten -f https://shi-labs.com/natten/wheels/cu116/torch1.13/index.html 
```

#### 1.2 Dataset

You can download the datasets you need from our [OneDrive](https://1drv.ms/u/s!AqKlMh-sml1mw362MfEjdr7orzds?e=budrUU) and place the downloaded datasets in the folder `datasets`. To use the YML profile we provide, keep the local folder `datasets` in the same directory tree as the OneDrive folder `datasets`.

| Task      | Dataset  | Relative Path                |
| --------- | -------- | ---------------------------- |
| SISR      | DF2K     | datasets/sr_data/DF2K        |
|           | Set5     | datasets/sr_data/Set5        |
|           | Set14    | datasets/sr_data/Set14       |
|           | BSD100   | datasets/sr_data/BSD100      |
|           | Urban100 | datasets/sr_data/Urban100    |
|           | Manga109 | datasets/sr_data/Manga109    |
| Denoising | SIDD     | datasets/denoising_data/SIDD |

> ???? All datasets have been processed in IMDB format and do not require any additional processing. The processing of the SISR dataset refers to the [BasicSR document](https://basicsr.readthedocs.io/en/latest/api/api_scripts.html), and the processing of the denoising dataset refers to the [NAFNet document](https://github.com/megvii-research/NAFNet/tree/main/docs).

> ???? To verify the integrity of your download, please refer to `docs/md5.txt`.

#### 1.3 Pretraining Weight
| Model                                        | Relative Path |
| -------------------------------------------- | ------------- |
| NatIR                                        | [experiments/NATIR_LSR_x4/models](https://github.com/qwopqwop200/NatIR/blob/main/experiments/NATIR_LSR_x4/models/net_g_490000.pth) |

### 2 Run

When running the FRL code, unlike BasicSR, you must specify two YML configuration files. The run command should be as follows:

```shell
python ${function.py} -expe_opt ${expe.yml} -task_opt ${task.yml}
```

- `${function.py}` is the function you want to run, e.g. `test.py`
- `${expe.yml}` is the path to the experimental YML configuration file that contains the model-related and training-related configuration, e.g. `options/repr/NATIR/NATIR_LSR.yml`
- `${task.yml}` is the path to the task YML configuration file that contains the task-related configuration, e.g. `expe/task/LSR_x4.yml`

> ???? A complete experiment consists of three parts: the data, the model, and the training strategy. This design allows their configuration to be decoupled. 

For your convenience, we provide a demo test set `datasets/demo_data/Demo_Set5` and a demo pre-training weight `experiments/NATIR_LSR_x4/models/net_g_490000.pth`. Use the following commands to try out the main functions of the FRL code.

#### 2.1 Train

This function will train a specified model.

```shell
python train.py -expe_opt options/repr/NATIR/NATIR_LSR.yml -task_opt options/task/LSR_x4.yml
```

![](figs/train.png)

> ???? Use the following demo command instead if you prefer to run in CPU mode:
>
> ```shell
> python train.py -expe_opt options/repr/NATIR/NATIR_LSR.yml -task_opt options/task/LSR_x4.yml --force_yml num_gpu=0
> ```

#### 2.2 Test

This function will test the performance of a specified model on a specified task.

```shell
python test.py -expe_opt options/repr/NATIR/test_NATIR.yml -task_opt options/task/LSR_x4.yml
```

![](figs/test.png)

#### 2.3 Infer

You can use this function to restore your own image.

```shell
python infer.py -expe_opt options/repr/NATIR/NATIR_LSR.yml -task_opt options/task/LSR_x4.yml
```

## Useful Features

### Data Flow

The image restoration process based on the BasicSR is as follows:

```
              ?????????????????????????????????                ?????????????????????????????????                 ?????????????????????????????????
              ???  image  ??????????????????????????????????????????????????????  model  ?????????????????????????????????????????????????????????  image  ???
              ????????????????????????????????? pre-processing ????????????????????????????????? post-processing ?????????????????????????????????
```

By default, the pre-processing operation normalizes the input image of any bit range (e.g., an 8-bit RGB image) to the [0, 1] range, and the post-processing operation restores the output image to the original bit range. The default data flow is shown below:

```
              ?????????????????????????????????                ?????????????????????????????????                 ?????????????????????????????????
              ??? 0-2^BIT ??????????????????????????????????????????????????????   0-1   ????????????????????????????????????????????????????????? 0-2^BIT ???
              ?????????????????????????????????                ?????????????????????????????????                 ?????????????????????????????????
```

However, for some input images (e.g., 16-bit TIF images), this data flow may lead to unstable training or degraded performance. Therefore, the FRL code provides support for data flows of any bit range. The new data flows are shown below:

```
              ?????????????????????????????????                ?????????????????????????????????                 ?????????????????????????????????
              ??? 0-2^BIT ?????????????????????????????????????????????????????? 0-2^bit ????????????????????????????????????????????????????????? 0-2^BIT ???
              ?????????????????????????????????                ?????????????????????????????????                 ?????????????????????????????????
```

You can try different data flows by simply changing the parameter `bit` in the file `${expe.yml}`. Set it to `0` to use the default data flow of BasicSR.

> ???? We tested the impact of different data flows on the SISR task (by retraining the EDSR, RFDN, and ELAN models using 8-bit RGB images). The results show that 8-bit models (trained with 8-bit data flow) perform **slightly better** than 0-bit models.
>

> ?????? We **did not** test the impact of different data flows on other image restoration tasks.

> ?????? Using new data flows may lead to **inaccurate** metric results (PSNR: error less than 0.001; SSIM: error less than 0.00001). To get more accurate metric results, use `scripts/evaluate.m` instead.

### LMDN Loading

A standard BasicSR LMDB database structure is as follows:

```
                                        demo.lmdb
                                        ????????? data.mdb
                                        ????????? lock.mdb
                                        ????????? meta_info.txt
```

By default, BasicSR automatically reads the file `demo.lmdb/meta_info.txt` when loading the LMDB database. In the FRL code, you can specify the file `meta_info.txt` to be used when loading the LMDB database. This makes it easier to process datasets, such as splitting a dataset into a training set and a test set.

> ???? The LMDB database of BasicSR has a unique form. More information about LMBD database and file `mateinfo.txt` can be found in the [BasicSR document](https://basicsr.readthedocs.io/en/latest/index.html).

### Model Customization

Different from BasicSR, all models in the FRL code **must** have the following four parameters:

- `upscale`: upscale factor, e.g: 2, 3, and 4 for lsr task, 1 for denoising task
- `num_in_ch`: input channel number
- `num_out_ch`: output channel number
- `task`: image restoration task, e.g: lsr, csr or denoising

A demo model implementation is as follows:

```python
import torch


class DemoModel(torch.nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,  # noqa
                 num_groups: int, num_blocks: int, *args, **kwargs) -> None:  # noqa
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

> ???? The FRL code automatically assigns values to these parameters based on the task configuration file used, so you do not need to define them in the parameter `network_g`.
>
> ```yaml
> network_g:
>    type: DemoModel
>    # Only the following parameters are required!
>    num_groups: 20
>    num_blocks: 10
> ```

## Out-Of-The-Box Models

Standing on the shoulders of giants allows us to grow quickly. So, we implemented many out-of-the-box image restoration models that may help your work. Please refer to the folder `archs` and the folder `options/expe` for more details.

You can use the following command **out of the box**!

```shell
# train EDSR to solve 2x classic super-resolution task
python train.py -expe_opt options/expe/EDSR/EDSR_CSR.yml -task_opt options/task/CSR_x2.yml

# test the performance of IMDN on 3x lightweight super-resolution task
python test.py -expe_opt options/expe/IMDN/IMDN_LSR.yml -task_opt options/task/LSR_x3.yml

# analyse the complexity of RFDN on 4x classic lightweight super-resolution task
python analyse.py -expe_opt options/expe/RFDN/RFDN_LSR.yml -task_opt options/task/LSR_x4.yml
```

We provide many experimental and task YML configuration files. To perform different experiments, feel **free** to combine them in the command.

> ???? If these implementations help your work, please consider citing them. Please refer to file `docs/third_party_works.bib` for more information.

# Acknowledgements
This code is based on [Fried Rice Lab](https://github.com/fried-rice-lab/friedricelab) and Neighborhood-Attention(https://github.com/SHI-Labs/NATTEN)

This code is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank its developers for creating such a useful toolbox. The code of the function analyse is based on [NTIRE2022 ESR](https://github.com/ofsoundof/NTIRE2022_ESR), and the code of the function interpret is based on [LAM](https://github.com/X-Lowlevel-Vision/LAM_Demo). All other image restoration model codes are from their official GitHub. More details can be found in their implementations.
