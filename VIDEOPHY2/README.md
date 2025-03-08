# VIDEOPHY-2

This repository contains the official implementation and data for "VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation".

[[Webpage]()] [[Paper]()] [[Test Dataset 🤗](https://huggingface.co/datasets/videophysics/videophy2_test)] [[Train Dataset 🤗](https://huggingface.co/datasets/videophysics/videophy2_train)] [[AutoRater 🤗](https://huggingface.co/videophysics/videophy2_autoeval/tree/main)] [[Twitter]()]

## Abstract
Large-scale video generative models, capable of creating realistic videos of diverse visual concepts, are strong candidates for general-purpose physical world simulators. However, their adherence to physical commonsense across real-world actions remains unclear (e.g., playing tennis, backflip). Existing benchmarks suffer from limitations such as limited size, lack of human evaluation, sim-to-real gaps, and absence of fine-grained physical rule analysis. To address this, we introduce VideoPhy-2, an action-centric dataset for evaluating physical commonsense in generated videos. We curate 200 diverse actions and detailed prompts for video synthesis from modern generative models. We perform human evaluation that assesses semantic adherence, physical commonsense, and grounding of physical rules in the generated videos. Our findings reveal major shortcomings, with even the best model achieving only 22% joint performance (i.e., high semantic and physical commonsense adherence) on the hard subset of VideoPhy-2. We find that the models particularly struggle with conservation laws like mass and momentum. Finally, we also train VideoPhy-2-AutoEval, an automatic evaluator for fast, reliable assessment on our dataset. Overall, VideoPhy-2 serves as a rigorous benchmark, exposing critical gaps in video generative models and guiding future research in physically-grounded video generation.

<h1 align="center"><img src="main_graph.png"></h1>
<h1 align="center"><img src="videophy2.png"></h1>

## Human LeaderBoard 🏆
We evaluate 7 closed and open text-to-video generative models on VideoPhy-2 dataset with **human annotation**. We report the joint performance. It is computed by aggregating the percentage of testing prompts for which the T2V models generate videos that adhere to the conditioning caption (SA>=4) and exhibit high physical commonsense (PC>=4). We also present a hard subset of the data, which is constructed using CogVideoX-5B as the reference model.

<div align="center">

| **#** | **Model** | **Source** | **All** | **Hard** | 
| -- | --- | --- | --- | --- | 
| 1      | [Wan2.1-T2V-14B]()🥇 | Open |  **39.6**  | **63.3**   | 
| 2 | [CogVideoX-5B](https://github.com/THUDM/CogVideo)🥈 | Open | 19.7 | 41.1 | 
| 3 | [Cosmos-Diffusion-7B](https://arxiv.org/abs/2401.09047)🥉 | Open |19.0 | 48.5 | 
| 4 | []() | Open |18.6 | 47.2 |
| 5 | []() | Open | 15.7 | 48.7 | 
| 6 | []() | Closed  | 13.6   | 61.9   |
| 7 | [VideoCrafter2]() | Closed | 12.5 | 48.5 | 

We manually evaluate OpenAI Sora on a subset of 60 promptss using its UI due to the lack of an API.

</div>

## VideoPhy-2-AutoEval


### Installation

1. Use the same instructions as the VideoPhy-1 [README](release/videophy/README.md).

2. In this work, we propose an auto-evaluator for our dataset. 

The model checkpoint is publicly available on [🤗 Model](https://huggingface.co/videophysics/videophy2_autoeval/tree/main).

### Inference

1. Download the model checkpoint to your local machine. 
```python
git lfs install
git clone https://huggingface.co/videophysics/videophy2_autoeval
```
2. Since this model was trained in a multi-task setting, it can be used for semantic adherence judgment, physical commonsense judgment, and physical rule classification.

```
    Semantic Adherence: Outputs a score that is one of 1-5
    Physical Commonsense: Outputs a score that is one of 1-5
    Physical Rule: Classifies whether a particular physical rule is grounded in the generated video.
```

3. 

4. 

5. 

### Training VideoCon-Physics

#### Data

1. We release the training data on Huggingface dataset - [train_dataset](https://huggingface.co/datasets/videophysics/videophy_train_public).
2. For training, use the same instructions as highlighted in the VideoPhy-1 [README](https://github.com/Hritikbansal/videophy?tab=readme-ov-file#training). 
3. One critical thing to note is that VideoPhy-2-AutoEval trains the [VideoPhy-1 auto-rater](https://huggingface.co/videophysics/videocon_physics/tree/main) instead of VideoCon. 

### Citation
```

```
