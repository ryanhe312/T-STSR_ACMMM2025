# T-STSR

Official Implementation of "Efficient Trajectory Space-Time Super-Resolution for Fast Live-cell Imaging" (ACMMM 2025). 

## Environment

We use Pytorch 1.13.1 and CUDA 11.7 for this project, please run the following commands to set up the environment.

```bash
pip install -r requirements.txt
```

## Dataset

We synthesize data following [Particle Tracking Challenge](https://www.nature.com/articles/nmeth.2808). We use real-world datasets from [CCR5](https://elifesciences.org/articles/76281) and [Lysosome](https://www.nature.com/articles/s41592-023-02138-w). Please download the processed data from the [Release](https://github.com/ryanhe312/T-STSR_ACMMM2025/releases) page and extract the files here. 

## Benchmark

Models can be downloads at [Release](https://github.com/ryanhe312/T-STSR_ACMMM2025/releases) page. Please run the following commands to experiment with synthetic data.

```bash
bash eval_synthetic.sh
```

Please run the following commands to experiment with real-world data.

```bash
bash eval_bioimaging.sh
```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{he2025efficient,
  title={Efficient Online Training for Zero-Shot Time-Lapse Microscopy Denoising and Super-Resolution},
  author={He, Ruian and Zhang, Zixian and Cheng, Ri and Tan, Weimin and Yan, Bo},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025}
}
```