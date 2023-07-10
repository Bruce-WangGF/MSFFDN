# MSF$^2$DN: Mutil Scale Feature Fusion Dehazing Network with Dense Connection (ACCV2022)

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

This is the official PyTorch codes for the paper.  

>**MSF$^2$DN: Mutil Scale Feature Fusion Dehazing Network with Dense Connection**<br> Guangfa Wang, Xiaokang Yu*  (* indicates corresponding author)<br>
>The 16th Asian Conference on Computer Vision， ACCV2022

<img src="E:\研三\ACCV相关文件\ACCV需要文件\LaTeX2e Proceedings Templates download\image\Network_Architecture\NA\NA_00.jpg" alt="NA_00" style="zoom:70%;" />



## Demo

<img src="E:\研三\ACCV相关文件\ACCV需要文件\LaTeX2e Proceedings Templates download\image\real\Input1.jpg" alt="Input1" style="zoom:80%;" /><img src="E:\研三\ACCV相关文件\ACCV需要文件\LaTeX2e Proceedings Templates download\image\real\Ours1.jpg" alt="GRID1" style="zoom:80%;" />



## Get Started

### Prepare models

Downloading checkpoints

<table>
<thead>
<tr>
    <th>Model</th>
    <th> Description </th>
    <th>:link: Download Links </th>
</tr>
</thead>
<tbody>
<tr>
    <td>MSF$^2$DN</td>
    <th>Mutil Scale Feature Fusion Dehazing Network with Dense Connection</th>
    <th rowspan="3">
    [<a href="https://pan.baidu.com/s/1POs2MEu5FKF16RQ9AYTDNA">Baidu Disk (pwd: 5z34)</a>]
    </th>
</tr>
</tbody>
</table>

### Quick demo

Run demos to process the images in dir `./text_image/` by following command, before this, you should input the right path in dehaze.py:

```
python dehaze.py
```

### Train MSF$^2$DN

Train our MSF$^2$DN, before this, you should input the right path in the corresponding files, especially in the dataloader.py, and then run the following command:

```
python train.py
```

## Citation

If you find our repo useful for your research, please cite us:

```
@InProceedings{Wang2022ACCV,
    author    = {Wang, Guangfa and Yu, Xiaokang},
    title     = {MSF\${\textasciicircum}2\$DN:Multi Scale Feature Fusion Dehazing  Network with Dense connection},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {2950-2966}
}

```

## Acknowledgement

This repository is maintained by Guangfa Wang.