

# Implement the model with hard attention

## Hard attention modules

There are two ways to index the hard attention modules, with experience ID or class ID. 


### Indexing with experience ID

Different from ordinary continual learning scenarios, different experiences could contain the same classes of images. Therefore, during the inference, it is not clear that which experience IDs should be used to index the hard attention modules.

One way to deal with this use the experience IDs that the classes last appeared in. For instance, if a class appears in experience `0`, `15`, and `30`, the we use the experience ID `30` to index the hard attention module in order to get the logits for this class. Note that logits must be generated individually for each class in this case.


### Indexing with class ID

It makes more sense semantically to index the hard attention modules with class IDs, as the information that we would like the network to retain is class-specific.

The biggest problem with this approach is that the number of classes is too much for the given network. This will limit the learning of classes that appear later in the sequence of experiences.

To implement this, we have to modify the dataloader so that the batched images share the same class IDs (positive & negative). This is because the hard attention modules cannot accept different indices in a single forward/backward pass. 

