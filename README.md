# Cardiac Echonet Demo Powered by OpenVINO

This project demonstrates OpenVINO optimized EchoNet-Dynamic. EchoNet-Dynamic is an end-to-end beat-to-beat deep learning model for semantic segmentation of the left ventricle, prediction of ejection fraction, and assessment of cardiomyopathy with reduced ejection fraction.

## Create and activate isolated Python environment

```
conda create -y -n echonet python=3.10
```

```
conda activate echonet
```

```
pip install opencv-python
```

```
git clone https://github.com/gsilva2016/dynamic.git
cd dynamic
pip install .
cd ..
```

```
pip install -r requirements.txt
```

## Videos

Ensure videos for inference are in the videos folders

## Download open-source models

Download segmentation model with wget

```
wget https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt
```

or using curl

```
curl -L -o deeplabv3_resnet50_random.pt https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt
```

Download ejection fraction model with wget

```
wget https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt
```

or using curl

```
curl -L -o r2plus1d_18_32_2_pretrained.pt https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt
```

## Convert the finetuned open-source models to optimized OpenVINO models

```
python convert_models.py --ef_model r2plus1d_18_32_2_pretrained.pt --seg_model deeplabv3_resnet50_random.pt
```

## Start the Demo

Execute the below command to excute only the segmentation workload using iGPU. Note device NPU is not supported at this time.

```
python external_inference.py --videos_dir videos --output_dir .\output\external_inference --seg_model seg.xml --display --display_loops 5 --device GPU
```

Execute the below command to excute both ejection fraction and segmentation workloads using iGPU

```
python external_inference.py --videos_dir videos --output_dir .\output\external_inference --ef_weights .\models\r2plus1d_18_32_2_pretrained.pt --seg_weights .\models\deeplabv3_resnet50_random.pt --device GPU
```