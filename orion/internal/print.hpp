//
// Created by macross on 9/17/22.
//

#ifndef ORION_PRINT_HPP
#define ORION_PRINT_HPP

#include <iostream>

namespace orion
{

// prints a rank 3 tensor with batch/channels as last dimension
void Print(const Tensor<3> &tensor)
{
    std::cout << "[";
    for (Eigen::Index i = 0; i < tensor.dimension(2); i++) {
        Eigen::array<Eigen::Index, 3> offset{0, 0, i};
        Eigen::array<Eigen::Index, 3> extent{tensor.dimension(0),
                                             tensor.dimension(1), 1};
        std::cout << tensor.slice(offset, extent).reshape(Tensor<2>::Dimensions(tensor.dimension(0), tensor.dimension(1))) << "]\n\n";
    }
}

}

#endif //ORION_PRINT_HPP
