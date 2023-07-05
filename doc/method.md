Our method combines the strengths of Hard Attention to the Task (HAT) and Supervised Contrastive Learning, both tailored to the Class-incremental with Repetition (CIR) setting.

Essentially, we partition the network based on the experience ID using trainable hard (binary) masks. This enables us to selectively update only the network's parameters associated with the current experience during backpropagation, mitigating catastrophic forgetting.

The original HAT suffers from the cases where the number of experiences is large. To address this issue, we propose a different way to initialize the hard attention masks and align the gradients, so that the training process is more stable and closer to non-HAT training. 

## Single Model Training

Training for each experience is divided into two stages:

1. Representation learning: Supervised Contrastive Learning is utilized to learn a good data representation where images from the same classes are close and images from different classes are far away from each other. 
2. Classification learning: During this phase, we focus only on the classes present in the current experience for classification. An additional logit is introduced to the network for all replayed classes. Normalization is applied to scale the logits, ensuring that logits from different experiences share a consistent scale.

Our single model training produces an averaged accuracy of 40.19% on configuration 1, 2 and 3.


## Training with replicas

The training process for each experience is the same as the single model training. The only difference is how model replicas are assigned to different experiences. We have two approaches:
1. Different models for different experiences (fragments): In this case, each model is trained on roughly equal number of experiences. For instance, given 10 experiences and 5 models, each model is trained on 2 experiences, i.e., the first model is trained on experiences 0 and 1, the second on 2 and 3, and so on. Even though models' weights are not shared, they are initialized from the preceding model, if available, promoting knowledge transfer.
2. Different models on the same experiences (ensembles). This strategy complements the first. For each experience, we train a set of M independent models simultaneously.

Here are the results for the two approaches on configuration 5:

| # fragments |  # ensembles | Accuracy |  Accuracy change |
|-------------|--------------|----------|------------------|
|      1      |      1       |  42.77%  | baseline         |
|     10      |      1       |  48.20%  | + 5.43%          |
|     25      |      1       |  55.74%  | +12.97%          |
|     50      |      1       |  65.57%  | +22.80%          |
|     50      |      2       |  68.04%  | +25.27%          |

## Testing

Given that the experience ID forms part of the input for HAT networks, it's necessary to test the model across multiple experiences to retrieve logits for all classes. To augment predictions, we also incorporate logits from earlier experiences. As a result, each class has multiple logits from different experiences. We compute the final prediction by taking the weighted average of these logits.
