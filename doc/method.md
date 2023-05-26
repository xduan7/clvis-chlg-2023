Our method is based on HAT (hard attention to the task). 

Essentially, the hard attention partitions the network into different segments, reserved for different experiences, and therefore prevents catastrophic forgetting.
For each experience, our training consists of two phases: (1) representation learning phase using supervised contrastive learning, and (2) classification learning phase, where the network only learns the current classes in the experience, with an additional logit for out-of-experience classification (for replayed embeddings).

During the prediction, for each class, we use the logits from the last experience(s) where the class is seen.
The network automatically preserves the momentum from the last experience(s) to further improve performance.

Without any buffer, module list containing weight parameters, or anything else for model replicas, our method achieves an average accuracy of 41.2% on configurations 1, 2, and 3 with the total number of 5,338,604 parameters, which yields a 5.863MB model (takes 1200MB GPU memory during training).


To run the model
```bash
python train.py --cuda 0 --config_file config_s1.pkl --run_name c1 --hat --hat_grad_comp_factor 100 --hat_reg_decay_exp 0.5 --hat_reg_enrich_ratio -1.4 --rep_num_epochs 28 --rep_lr 0.0072 --rep_batch_size 64 --rep_hat_reg_base_factor 1.3 --rep_proj_head_dim 256 --rep_num_replay_samples_per_batch 32 --rep_proj_div_factor 0.025 --clf_num_epochs 48 --clf_lr 0.00085 --clf_batch_size 32 --clf_hat_reg_base_factor 2.0 --clf_freeze_hat --clf_num_replay_samples_per_batch 16 --clf_logit_reg_factor 0.02 --clf_logit_reg_degree 2 --clf_logit_calibr batchnorm --clf_train_exp_logits_only --tst_time_aug 18 --clf_use_momentum
```

The training with replicas takes about 300 minutes on NVIDIA V100. 
The hyperparameters slightly different from the ones we used during the first phase of the challenge.


We have not realized that replicas of models are allowed since it essentially changed the number of trainable parameters. However, we made some last-minute changes and added options for model replicas.

There are two types of replicas, (1) is to fragment the model into N different versions, which will handle different experiences separately; (2) is the ensemble of the models: there are multiple models for the same experience, and their predictive results will be averaged together during the test.

Both options seem to increase the performance.

To run the model with replicas (50 fragments, 2 ensembles)
```bash
python train.py --cuda 0 --config_file config_s1.pkl --run_name c1 --hat --hat_num_fragments 50 --hat_num_ensembles 2 --hat_grad_comp_factor 100 --hat_reg_decay_exp 0.5 --hat_reg_enrich_ratio -1.4 --rep_num_epochs 28 --rep_lr 0.0072 --rep_batch_size 64 --rep_hat_reg_base_factor 1.3 --rep_proj_head_dim 256 --rep_num_replay_samples_per_batch 32 --rep_proj_div_factor 0.025 --clf_num_epochs 48 --clf_lr 0.00085 --clf_batch_size 32 --clf_hat_reg_base_factor 2.0 --clf_freeze_hat --clf_num_replay_samples_per_batch 16 --clf_logit_reg_factor 0.02 --clf_logit_reg_degree 2 --clf_logit_calibr batchnorm --clf_train_exp_logits_only --tst_time_aug 18 --clf_use_momentum --num
```

The training with replicas takes about 450 minutes on NVIDIA V100.
