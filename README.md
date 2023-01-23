# Automatic_Signal_Detector

<p align="center">
   <img src="images/Signal_Language.png" width="40%" height="40%"/>
</p>

## Abstract
In this project, there are two parts:
- Create hands dataset
- Train model to recognize hands (Here, implemented three models: MLP, CNN, Transfer Learning)

In order to create hands dataset, we need to find a way to detect our hands. There are some pattern detecting faces and eyes. Given that patterns, we use Camshift algorithm to detect hands by setting face region probability to zero, and then it's likely that hands is the most likely region to be detected.

## Getting Started
Clone the repository.

```
git clone https://github.com/Zhijie-He/Automatic_Signal_Detector.git
```

Goto the cloned folder.
```
cd Automatic_Signal_Detector
```

## Steps to run Code

```
### For Linux Users
python3 -m venv psestenv
source psestenv/bin/activate

### For Window Users
python3 -m venv psestenv
cd psestenv
cd Scripts
activate
cd ..
cd ..
```

Install requirements with mentioned command below.

```
pip install -r requirements.txt
```


Create hands dataset

```bash
python create_hands_dataset [A-Z]
```
- `--[A-Z]` specifies the signal name.

Select the face capture by type "y"

After decide the face bounding, type "y" to create hand dataset according to the hands gesture(Signal_Language).


<p align="center">
   <img src="images/create_hand_dataset.png" width="40%" height="40%"/>
</p>

Train model to recoginze signal language, see model performance below
```bash
python main model_name
```
- `model_name` specifies the model name, can select from MLP, CNN and TF.

## Hands dataset example

<p align="center">
   <img src="images/dataset.png" width="40%" height="40%"/>
</p>


## Different ways to detect face

<table>
  <tr>
    <td>Common way to detect faces using Haar Cascades</td>
     <td>Optimization by seraching subregion </td>
     <td>MeanShift</td>
     <td>CamShift </td>
  </tr>
  <tr>
    <td><img src="images/tradition_way.gif" width=350 height=300></td>
    <td><img src="images/search_region.gif" width=350 height=300></td>
    <td><img src="images/meanShift.gif" width=350 height=300></td>
    <td><img src="images/CamShift.gif" width=350 height =300></td>
  </tr>
 </table>



## Model Performance


<h3>MLP</h3>
<img src="images/MLP_performance.png" width="100%" height="50%"/>

<h3>CNN</h3>

<img src="images/CNN_performance.png" width="100%" height="50%"/>

<h3>Transfer Learning</h3>
<img src="images/TF_performance.png" width="100%" height="50%"/>
