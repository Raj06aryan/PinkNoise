Install the conda environment using:

```bash
conda env create -f environment.yml
```

This will create a conda environment named `pink` with all the required dependencies.
Activate the environment using:

```bash
conda activate pink
```
You might face the following errors while running the examples:

- `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
    - go to your site-package directory "/home/<user>/miniconda3/envs/pink/lib/python3.9/site-packages/pink/sb3.py"
    - add the line `device = th.device("cuda" if th.cuda.is_available() else "cpu")` at the top of the file
    - add the line `cn_sample = cn_sample.to(device)` below line 123

You can now run the examples in the `examples` directory.