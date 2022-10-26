//
// Created by macross on 8/30/22.
//

#ifndef ORION_BATCHER_HPP
#define ORION_BATCHER_HPP

#include "orion/orion_types.hpp"
#include <vector>

namespace orion
{

std::vector<Tensor<2>>
VectorToBatch(std::vector<std::vector<Scalar>> &dataset, int batch_size);

template<typename DataType>
std::vector<std::vector<DataType>>
CSVToVector(const std::string &filename, char delimiter = ',',
            std::vector<int> skip_cols = {});

}


#endif //ORION_BATCHER_HPP
