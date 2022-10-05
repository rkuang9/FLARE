//
// Created by macross on 10/1/22.
//

#ifndef ORION_PRINT_TENSOR_HPP
#define ORION_PRINT_TENSOR_HPP

#include "orion/orion_types.hpp"

namespace orion
{

void PrintNHWCAsNCHW(const Tensor<4> &img_batch);


/**
 * Prints the result of extract_image_patches, using Row-Major tensor format,
 * in the order batch, patch, channel, width, height
 *
 * The last two dimensions will be displayed in 2D
 *
 * @param patches   Tensor<5> in format NPWHC (batch, patch, width, height, channel)
 */
void PrintPatches(const Tensor<5> &patches);

}

#endif //ORION_PRINT_TENSOR_HPP
