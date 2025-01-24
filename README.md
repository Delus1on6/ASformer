
1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```
1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with ASformer
bash ./scripts/ECL/ASformer.sh
bash ./scripts/Weather/ASformer.sh
bash ./scripts/Exchange/ASformer.sh
```
