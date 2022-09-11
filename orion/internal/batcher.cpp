//
// Created by macross on 8/30/22.
//

#include "batcher.hpp"
#include <iostream>

namespace orion::internal
{

std::vector<Tensor2D>
VectorToBatch(std::vector<std::vector<Scalar>> &dataset, int batch_size)
{
    int num_batches = dataset.size() / batch_size;
    int batch_remainder = dataset.size() % batch_size;
    int num_features = dataset.front().size();

    std::vector<Tensor2D> batched_dataset; // return value

    for (int i = 0; i < num_batches; i++) {
        Tensor2D batch(num_features, batch_size);

        // visit each vector element batch_size at a time to
        for (int j = i * batch_size; j < i * batch_size + batch_size; j++) {
            batch.chip(j % batch_size, 1) = Tensor1DMap(dataset[j].data(), num_features);
        }

        batched_dataset.push_back(std::move(batch));
    }

    // create a batch from the leftover dataset values that don't make a full batch
    if (batch_remainder > 0) {
        Tensor2D batch_leftover(num_features, batch_remainder);

        for (int m = 0; m < batch_remainder; m++) {
            batch_leftover.chip(m, 1) = Tensor1DMap(dataset[num_batches * batch_size + m].data(), num_features);
        }

        batched_dataset.emplace_back(std::move(batch_leftover));
    }

    return batched_dataset;
}

}