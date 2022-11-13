#include <iostream>
#include <chrono>
#include <vector>
#include <orion/orion.hpp>
//#include "examples/xor_classifier.hpp"

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

using namespace orion;


void maxpool_op()
{
    Tensor<4> img(2, 5, 5, 3);
    PoolSize pool(3, 3);
    Stride stride(1, 1);
    Eigen::Index batch = img.dimension(0);
    Eigen::Index gap_w = pool.width() - 1;
    Eigen::Index gap_h = pool.height() - 1;
    Eigen::Index img_w = img.dimension(2);
    Eigen::Index img_h = img.dimension(1);
    Eigen::Index pool_size = pool.TotalSize();
    Eigen::Index pool_h = pool.height();
    Eigen::Index pool_w = pool.width();
    Eigen::Index channels = img.dimension(3);

    for (int i = 0; i < img.size(); i++) {
        img(i) = i;
    }

    Tensor<4> gradients(2, 3, 3, 3);
    gradients.setRandom();

    Tensor<2> gradients_flattened = gradients.reshape(Dims<2>(
            gradients.dimension(0) * gradients.dimension(1) * gradients.dimension(2),
            gradients.dimension(3)));

    Tensor<2> flatten = img.reshape(Dims<2>(
            img.dimension(0) * img.dimension(1) * img.dimension(2),
            img.dimension(3)));
    std::cout << flatten << "\n\n";

    Tensor<2> input_grad(flatten.dimensions());
    input_grad.setZero();

    Eigen::Index output_h = 1 + (img.dimension(1) - pool.height()) / stride.height();
    Eigen::Index output_w = 1 + (img.dimension(2) - pool.width()) / stride.width();


    for (Eigen::Index i = 0; i < (output_h * output_w * batch); i++) {
        Eigen::Index patch_start =
                i + (i / output_w) * gap_w + (i / pool_size) * img_w * gap_h;
        //std::cout << "patch index start: " << patch_start << "\n";
        Scalar max_val = INT_MIN;
        Eigen::Index max_index = -1;

        for (Eigen::Index c = 0; c < channels; c++) {
            for (Eigen::Index p = 0; p < pool_size; p++) {
                Eigen::Index pool_index = patch_start + p + (p / pool_w) * gap_w;
                //std::cout << flatten(pool_index, c) << ", ";

                if (flatten(pool_index) > max_val) {
                    max_val = flatten(pool_index);
                    max_index = pool_index;
                }
            }

            input_grad(max_index, c) = gradients_flattened(i, c);

            std::cout << "\n";
        }

    }

    std::cout << "gradients\n" << gradients.shuffle(Dims<4>(3, 0, 1, 2)) << "\n";
    std::cout << "dL_dX:\n";
    std::cout << input_grad.reshape(img.dimensions()).shuffle(Dims<4>(3, 0, 1, 2));
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    maxpool_op();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}