(import 'polIter_deepseekSft2_dpo_positive_MATH_debug.jsonnet')
+
{
    episode_generator+: {
        dataset_portion: 1.0, # just change to full dataset
    }
}
+ (import 'trainers/64_samples.jsonnet')
+ (import 'episode_generators/dpo_positive/maxpair_64.jsonnet')
+ (import 'configs/trainers/dpo_b_0.1.jsonnet')
+ (import 'configs/trainers/3_epochs.jsonnet')