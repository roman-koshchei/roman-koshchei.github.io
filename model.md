# Model

Goal of object detection model is to be fine-tuned for our task and be runnable on edge device, which in my case is Rockchip SoCs: `rk3588` and `rk3576`.

Here are instructions on training model which fits my use-case. I don't want to pay licensing fees, so I moved from Ultralytics YOLO11 into PaddleDetection PP-YOLOE+ model.

To train/fine-tune model based on PP-YOLOE+ you first need to convert the dataset in COCO format, I use this structure:

```
|- annotations
|--- train.json
|--- valid.json
|- train
|--- {number}.png
|- valid
|--- {number}.png
```

## Installation

Clone the repository [github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) and install dependencies using next instructions. I use [UV package manager](https://docs.astral.sh/uv/getting-started/installation/), so install it before continuing.

Create Python environment with latest supported Python version of PaddleDetection, you can find it in official installation instructions: [/docs/tutorials/INSTALL.md](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/tutorials/INSTALL.md). For me:

```bash
uv venv --python 3.10
```

Then you need to install PaddlePaddle framework itself, go to [Quick install page](https://www.paddlepaddle.org.cn/en/install/quick) and find installation command for latest version for your hardware. At the moment for me it's:

```bash
uv pip install "paddlepaddle-gpu==3.1.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Install additional packages:

```bash
uv pip install setuptools
```

Create `check.py` script in top directory to check if PaddlePaddle installation was successful:

```python
import paddle
paddle.utils.run_check()
print(paddle.__version__)
```

Run like this, it may reveal that certain packages are required to be installed for proper functionality, so install those:

```bash
uv run check.py
```

Install dependencies of PaddleDetection and run `setup.py` script with `install` command:

```bash
uv pip install -r requirements.txt
```

```bash
uv run setup.py install
```

Install additional packages as well:

```bash
uv pip install scikit-learn "numba==0.56.4"
```

Run next test to ensure everything is properly installed, it may reveal additional packages required for installation:

```bash
uv run ppdet/modeling/tests/test_architectures.py
``` 

## Training

Put your dataset in `/dataset/{model}` folder. In `/configs/datasets` copy `coco_detection.yaml` file and change paths and directories to the ones of your dataset. Then go to `/configs/ppyoloe`, copy `ppyoloe_plus_crn_s_80e_coco.yml` and change dataset it's based on.

To start training process run:

```bash
uv run tools/train.py -c configs/ppyolo/{model}.yml --eval
```

Now wait until training is finished. If you interrupt the training, don't worry, you can resume training from checkpoint:

```bash
uv run tools/train.py -c configs/ppyolo/{model}.yml -r r output/{epoch} --eval
```

## Export

We need to export model for inference before exporting to ONNX, it's like cleanup step as far as I understand. We also exclude post processing and NMS, it will be done by CPU code. Run:

```bash
uv run tools/export_model.py -c configs/ppyoloe/{model}.yml -o weights={model-weights}.pdparams TestReader.inputs_def.image_shape=[3,640,640] TestReader.batch_size=1 exclude_nms=True trt=True exclude_post_process=True
```

In result you should get model exported folder `output_inference/{model}`. Then we need to install dependencies to export model to ONNX. Run:

```bash
uv pip install paddle2onnx
```

Time to export inference model to ONNX:

```bash
uv run paddle2onnx --model_dir output_inference/{model} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file {model}.onnx
```

As the result you will find `{model}.onnx` file in the root directory.

#TODO Run ONNX simplify to simplify the model and remove unnecessary operations.



