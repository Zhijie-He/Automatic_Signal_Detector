# Automatic_Signal_Detector
<img src="images/Signal_Language.png" width="40%" height="40%"/>

## Abstract
In this project, there are two parts:
- create hands dataset
- train model to recognize hands (Here, implemented three models: MLP, CNN, Transfer Learning)

## Getting Started
Get code: `git clone https://github.com/Zhijie-He/Automatic_Signal_Detector.git`

## Running the code

Create hands dataset

```bash
python create_hands_dataset [A-Z]
```
- `--[A-Z]` specifies the signal name.

Select the face capture by type "y"
After decide the face bounding, type "y" to create hand dataset according to the hands gesture(Signal_Language).
<img src="images/create_hand_dataset.png" width="40%" height="40%"/>

Train model to recoginze
```bash
python main model_name
```
- `model_name` specifies the model name, can select from MLP, CNN and TF.

## Hands dataset example

<img src="images/dataset.png" width="40%" height="40%"/>

## Different ways to detect face

### Common way to detect faces using Haar Cascades
<img src="images/tradition_way.gif" width="50%" height="50%"/>


### Optimization by seraching subregion 
<img src="images/search_region.gif" width="50%" height="50%"/>


### MeanShift
<img src="images/meanShift.gif" width="50%" height="50%"/>


### CamShift 
<img src="images/CamShift.gif" width="50%" height="50%"/>

## Model Performance

<h3>MLP</h3>
<img src="images/MLP_performance.png" width="100%" height="50%"/>

<h3>CNN</h3>

<img src="images/CNN_performance.png" width="100%" height="50%"/>

<h3>Transfer Learning</h3>
<img src="images/TF_performance.png" width="100%" height="50%"/>
