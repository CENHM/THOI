# Text-based Hand-object Interaction

Original implimentation: https://github.com/JunukCha/Text2HOI

# 1. Installation

Create an environment and activate it.

```bash
conda create -n $YOUR_ENV_NAME python=3.9
conda activate $YOUR_ENV_NAME
```

Dependencies:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install numpy==1.23
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib
pip install tqdm
$ pip install PyWavefront
pip install pymeshlab
pip install open3d
```

## 1.1. CLIP dependencies

Install additional dependencies and official [CLIP](https://github.com/openai/CLIP) repo as a Python package.
```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

## 1.2. MANO model and code

We use [MANO](https://mano.is.tue.mpg.de/) model and MANO inplimentation in [smplx](https://github.com/vchoutas/smplx) package, along with some part of Taheri's [code](https://github.com/otaheri/MANO). Note that you should follow the licenses of each repository you download.

- Download models (Models & Code) from the [MANO](https://mano.is.tue.mpg.de/) website.
- Unzip and copy the MANO models folder `.../mano_v*/models` into `.../Text2HOI/models/components/mano`
- Install additional dependencies.
  ```bash
  pip install smplx
  pip install trimesh==4.5.2
  pip install chumpy==0.70
  pip install pyglet==1.5.22
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
            |-- utils.py
```

# 2. Before running

There are a few things that need to be made before running the code. Please follow the instructions below:

- After you install `smplx` package, go to `.../Lib/site-packages/smplx/body_models.py -> MANO -> forward()`, and uncomment the following statement in the code, then the number of returning joint will be 16 (without 5 tips) instead of 21:
  ```python
  # Add pre-selected extra joints that might be needed
  joints = self.vertex_joint_selector(vertices, joints) # Uncomment
  ```
<!-- (*Optional*) Go to `.../Lib/site-packages/smplx/body_models.py -> SMPL -> __init__()`, and comment the following statement in the code:  
  ```python
  if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
  #    print(f'WARNING: You are using a {self.name()} model, with only'
  #          ' 10 shape coefficients.')
      num_betas = min(num_betas, 10)
  else:
      num_betas = min(num_betas, self.SHAPE_SPACE_DIM)
  ``` -->
- After you install `pytorch` package, go to `.../Lib/site-packages/torch/nn/functional.py -> _verify_batch_size()`, and comment the following statement in the code to allow situation when there are only one element in current batch:
  ```python
  size_prods = size[0]
  for i in range(len(size) - 2):
      size_prods *= size[i + 2]
    
  # Comment the following code:
  # if size_prods == 1:
  #     raise ValueError(
  #         f"Expected more than 1 value per channel when training, got input size {size}"
  #     )
  ```
- After you accuire GRAB dataset and finish the extraction using the official script, run `.../datasets/preprocess_grab_object_mesh.py` to preprocess the GRAB dataset (to reduce the number of vertices).

# 3. Run

There are three ways to run the porject:

1. Training without resuming from a checkpoint, and load your configuration in `.yml` file:
   ```bash
   python run.py --config_dir $YOUR_CONFIG_DIR 
   ```
2. Training with resuming from a checkpoint. In this mode, loading configuration from `--config_dir` is disabled:
   ```bash
   python run.py --resume [--checkpoint_dir $YOUR_CHECKPOINT_DIR] 
   ```
3. Inferencing. In this mode, loading configuration from `--config_dir` and `--resume` is disabled:
   ```bash
   python run.py --inference [--checkpoint_dir $YOUR_CHECKPOINT_DIR] [--result_path $YOUR_RESULT_DIR]
   ```