(import 'base.jsonnet') + {
    zero_optimization: {
        stage: 1,
        allgather_partitions: true,
        allgather_bucket_size: 5e8,
        overlap_comm: true,
        reduce_scatter: true,
        reduce_bucket_size: 'auto',
        contiguous_gradients: true,
    },
}
