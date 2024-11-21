# Text-based Hand-object Interaction

Original implimentation: https://github.com/JunukCha/Text2HOI

# 1. Installation

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

## 1.1. CLIP dependencies

Install additional dependencies and official [CLIP](https://github.com/openai/CLIP) repo as a Python package.
```
$ pip install ftfy regex
$ pip install git+https://github.com/openai/CLIP.git
```

## 1.2. MANO model and code

We use [MANO](https://mano.is.tue.mpg.de/) model and some part of Taheri's [code](https://github.com/otaheri/MANO). Note that you should follow the licenses of each repository you download.

- Download models (Models & Code) from the [MANO](https://mano.is.tue.mpg.de/) website.
- Unzip and copy the MANO models folder `.../mano_v*/models` into `.../Text2HOI/models/components/mano`
- Download Taheri's [code](https://github.com/otaheri/MANO), copy the `mano` folder into `.../Text2HOI/models/components/mano` 
- Install additional dependencies.
  ```
  $ pip install trimesh==4.5.2
  $ pip install chumpy==0.70
  $ pip install pyglet==1.5.22
  ```

Your folder structure should look like this:
```
THOI
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