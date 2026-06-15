# Relation Coding with Dot-Product Attention
This repository contains the implementation of an approach to encode binary entity relations in a way that dot-product attention can directly make use of. It works by defining one attention head per relation. For each head, the approach boosts attention scores of pairs that are related and suppresses scores of pairs that are not related. This emphasis step may be repeated to achieve more confident encoding.

The approach can, for example, encode relative token positions at the input level, avoiding the need to handle position labels in later model layers, reducing architectural restrictions due to dot-product attention's permutation invariance. It also simplifies the implementation of cache mechanisms such as [attention sinks][xiao2023streamingllm].

The following section presents a toy example to illustrate application parameters.


## Toy Example
The example [script][toyexamplesource] samples random token sequences from a fixed, randomly initialised vocabulary. It then trains a relation encoding/decoding model to reconstruct relation labels that were randomly masked at `50%`, while the masked token pairs are completely blocked from mixing during encoding.

We compare several setups to illustrate their behaviour. They are labelled `[v,a]+` where `v` is the number of value-only emphasis iterations, i.e. iterations without keys and queries, since mixing is completely based on relation labels. `a` is the number of full attention-based mixing iterations, in which the labels determine which scores are boosted or suppressed. The `+` indicates whether only true positives were boosted, which matches standard attention implementations. If `+` is omitted,  false positives are also actively suppressed by allowing negative value mixing weights.


### Encoding and Denoising Effectiveness
Top-1 Accuracy during training was as follows:

![Training Top-1 Accuracy](https://github.com/tim3237/experiments/blob/relcode/test_output/train_top1.png?raw=true)

It appears that `+` setups require more iterations as well as pure `v` setups compared to pure `a`. However, since all relation farther than `3` steps away were merged into a `>3` class for each direction, a model could already achieve `95.3%` accuracy by predicting only the extreme distance classes. The classifier was trained with higher weights for closer-distance relations, but these weights did not reflect the actual class imbalance. Therefore, we further inspect a random input/output example to see whether local relations are reconstructed reasonably.

![Example input, top-1 predictions, and d-top-1 predictions](https://github.com/tim3237/experiments/blob/relcode/test_output/train_sample.png?raw=true)

The first row shows the input sample, which was the same for all setups. Black relations were masked in labels and were also fully blocked during encoding, so the corresponding token embeddings could not mix directly. As a result, the model had to reconstruct them indirectly through other tokens available to both endpoints.

The second row shows that after three iterations, relations could be encoded effectively for denoised reconstruction. Among these setups, `[3,0]+` struggled the most with local relations, while far relations were still correctly reconstructed throughout. Except for `[0,1]`, candidates with only a single iteration made noticeable errors in these ranges.
The `d-top-1` predictions suggest that attention-based emphasis preserves distance-class coherence better than value-only mixing, since they tended to select the farthest available distance class for distant tokens.

To inspect local relations in more detail, we show centre crops below:

![Centre crops of train-length input-output samples.](https://github.com/tim3237/experiments/blob/relcode/test_output/train_crop.png?raw=true)


### Length Extrapolation
Since far relations are capped in our approach anyway, there is no hard limit on sequence length, and in principle we could increase it indefinitely without additional tricks. However, once we exceed training length, performance may drop if the approach does not extrapolate well in practice. The following plot shows `top-1` accuracy evaluated from training sample length of 128 (training length) up to a sample length of 1024:

![Evaluation Top-1 Accuracy](https://github.com/tim3237/experiments/blob/relcode/test_output/eval_top1.png?raw=true)

The plot suggests that attention-based setups handle length extrapolation well, especially those with fewer iterations. In contrast, value-only setups extrapolate better with more iterations. Among the positive-only variants, the setups with fewer iterations extrapolated slightly better. The mixed approach `[2,1]` shows similar extrapolation to `[3,0]`.

Again, because distances are capped, relation classes remain unevenly distributed and the imbalance becomes even more pronounced for longer sequences. At a sequence length of 1024, a predictor could already achieve `99.4%` accuracy by selecting only the extreme classes. As a result, slight accuracy decrease in accuracy during extrapolation may simply reflect changes in class distribution. The same applies to slight increases, which may be driven by strong performance on far-distance relations.

Thus, to better understand the dynamics, we inspect example outputs again:

![Example evaluation input](https://github.com/tim3237/experiments/blob/relcode/test_output/eval_sample.png?raw=true)

Centre crops:

![Centre crops of evaluation-length input-output samples](https://github.com/tim3237/experiments/blob/relcode/test_output/eval_crop.png?raw=true)

It shows that very far relations were easier for single-iteration attention setups, which helps explain their improved accuracy. Within local windows of six to eight tokens, all setups show consistent or even increasing performance. The size of this window is most likely determined by clamping distance relations to the bounds `[-4,...,4]`.

Again, `d-top-1` predictions suggest that distance-class coherence during length extrapolation is still better for attention-based setups.

These samples were not cherry-picked, so they should provide a reasonable impression, though minor details should not be over-interpreted.


## Parameter Choices
The toy example gives some indication of how effective the approach is for encoding positional information for later reconstruction. This suggests that the model may be able to represent position information in detail. However, while local context is often considered most important, it remains unclear which parts of positional information are most relevant for language modelling. Therefore, the toy example results should be interpreted with care.

The `a=3` setups showed strong performance, but they are also most computationally demanding. By contrast, `a=0` but `v=3` might be a good choice, since they do not require query or key layers or attention-score computation. However, the toy example does not measure how much the embeddings changed during encoding. Value-only mixing cannot take the required amount of change into account in the same way as attention-based mixing.

Finally, `[0,3]+` could likely be optimised efficiently using [Flex Attention][dong2024flex], whereas this is more difficult for `[0,3]` without post-softmax score modifications to support negative mixing weights.
The best parameter choice for practical models requires additional experiments.

[toyexamplesource]: https://github.com/tim3237/experiments/blob/relcode/test.py
[xiao2023streamingllm]: https://arxiv.org/abs/2309.17453v3
[dong2024flex]: https://arxiv.org/abs/2412.05496

