# AntiGAN

Running a GAN in reverse to do steganography. This was a project for the Scalable Machine Learning & Deep Learning course at KTH.

## How to run it

First, you'll need to install [Poetry](https://python-poetry.org/) and run `poetry install` within the repo to create a virtual environment which replicates our setup. After that, run `poetry shell` to run a shell within that environment.

We used [Guild](https://guild.ai/) for experiment tracking, and you can use it to replicate our results. Guild will already be installed within the virtual environment created by Poetry. Just do `guild run <operation> [parameters]` to run a single step of the pipeline. If all you want is to get everything to work with the default parameters, do:

```sh
guild run generate-examples
guild run train-reconstructor num_batches=1024 batch_size=16
guild run app
```
