# Text2HOI Replication

Original implimentation: https://github.com/JunukCha/Text2HOI

# Installation

Create an environment and activate it.

```
conda create -n <env_name> python=3.9
conda activate <env_name>
```

Dependencies:

```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
$ pip install numpy==1.23
$ pip install opencv-python
$ pip install scikit-image
$ pip install tensorboard
$ pip install matplotlib
$ pip install tqdm
```

## CLIP dependencies

Install additional dependencies and official [CLIP](https://github.com/openai/CLIP) repo as a Python package.
```
$ pip install ftfy regex
$ pip install git+https://github.com/openai/CLIP.git
```

## MANO model and code

We use [MANO](https://mano.is.tue.mpg.de/) model and some part of Taheri's [code](https://github.com/otaheri/MANO). Note that you should follow the licenses of each repository you download.

- Download models (Models & Code) from the [MANO](https://mano.is.tue.mpg.de/) website.
- Unzip and copy the MANO models folder `.../mano_v*/models` into `.../Text2HOI/models/components/mano`
- Download Taheri's [code](https://github.com/otaheri/MANO), copy the `mano` folder into `.../Text2HOI/models/components/mano` 
- Install additional dependencies.
  ```
  $ pip install trimesh
  $ pip install chumpy
  ```

Your folder structure should look like this:
```
Text2HOI
|-- models
    |-- components
        |-- mano
            |-- models
            |   |-- MANO_LEFT.pkl
            |   |-- MANO_RIGHT.pkl
            |   |-- ...
            |-- mano
                |-- joints_info.py
                |-- lbs.py
                |-- ...
```