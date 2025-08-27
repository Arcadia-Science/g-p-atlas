# Demo dataset

This directory contains a demo dataset for the G-P Atlas model. It is a very small toy dataset with 100 samples, 30 loci,and 3 phenotypes.

The dataset was generated using the `gpatlas.simulations` module as follows:

```bash
uv run -m gpatlas.simulations \
    --config-filepath simulation-config.yaml \
    --output-dirpath demo/dataset
```

The dataset is stored in the `demo/dataset` directory.

To train a model on this dataset, use the following command:

```bash
gpatlas train \
    --dataset demo/dataset \
    --output demo/output \
    --model-config model-config.yaml \
    --training-config train-config.yaml
```

The number of epochs in the training config is set to three to keep the training time short, even on a CPU. For more realistic datasets, the number of epochs should be increased.
