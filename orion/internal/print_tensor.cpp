//
// Created by macross on 10/1/22.
//

#include "print_tensor.hpp"
#include <iostream>

namespace orion
{

// NHWC(batch, height, width, channel) to NCHW(batch, channels, row, col)
void PrintNHWCAsNCHW(const Tensor<4> &img_batch)
{
    for (Eigen::Index batch = 0; batch < img_batch.dimension(0); batch++) {
        std::cout << "******* batch " << batch << " *******\n";
        Tensor<3> hwc = img_batch.chip(batch, 0);

        for (Eigen::Index channel = 0; channel < hwc.dimension(2); channel++) {
            std::cout << "------- channel " << channel << " ------- \n";
            std::cout << hwc.chip(channel, 2);
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}


// expects (batch, patch, width, height, channel)
// print as (batch, patch, channel, width, height)
void PrintPatches(const Tensor<5> &patches)
{
    std::cout << patches.shuffle(Tensor<5>::Dimensions(0, 1, 4, 2, 3)) << "\n\n";
    return;
    for (Eigen::Index batch = 0; batch < patches.dimension(0); batch++) {
        std::cout << "******* batch " << batch << " *******\n";
        Tensor<4> img_patch = patches.chip(batch, 0);

        for (Eigen::Index patch = 0; patch < img_patch.dimension(0); patch++) {
            std::cout << "------- patch " << patch << " ------- \n";
            Tensor<3> img = img_patch.chip(patch, 0);

            std::cout << img.shuffle(Tensor<3>::Dimensions(2, 0, 1)) << "\n";

            /*for (Eigen::Index channel = 0; channel < img.dimension(2); channel++) {
                std::cout << "------- channel " << channel << " ------- \n";
                Tensor<2> matrix = img.chip(channel, 2);
                std::cout << matrix << "\n";
            }*/
        }
    }
    std::cout << "\n";
}

}