#include <iostream>
#include <chrono>

#include <orion/orion.hpp>

using namespace orion;


/**
 * RowMajor expects NWHC (batch, width, height, channels)
 * Need data format NHWC (batch, height, width, channels)
 */
void im2col_convolution()
{
    int batch = 5;
    int channels = 3;
    int filters = 10;
    int filter_size = 2;
    int stride = 1;
    int output_w_h = 2;

    Tensor<3> preimage(1, 3, 3);
    preimage.setValues({{{1, 1, 3}, {1, 1, 2}, {2, 4, 0}}});
    Tensor<3> prekernel(1, 2, 2);
    prekernel.setValues({{{1, 0}, {2, 1}}});

    Tensor<4> image = preimage.reshape(Dims<4>(1, 3, 3, 1)).broadcast(
            Dims<4>(batch, 1, 1, channels));
    Tensor<4> kernel = prekernel.reshape(
            Dims<4>(1, filter_size, filter_size, 1)).broadcast(
            Dims<4>(filters, 1, 1, channels));

    std::cout << "kernel:\n" << kernel.shuffle(Dims<4>(0, 3, 1, 2))
            << "\nkernel dims: " << kernel.dimensions() << "\n\n";

    // for row-major, extract_image_patches returns (NPWHC), we shuffle it back to NPCWH where P = #patches
    Tensor<5> patches = image
            .extract_image_patches(filter_size, filter_size, stride, stride, stride,
                                   stride, Eigen::PADDING_VALID);

    std::cout << patches.shuffle(Dims<5>(0, 1, 4, 2, 3)) << "\npatches dims: "
            << patches.dimensions() << "\n\n";

    Eigen::Index patch_count = patches.dimension(1);


    Tensor<3> patches_im2col = patches.reshape(
            Dims<3>(batch, patch_count, filter_size * filter_size * channels));
    std::cout << "patches_im2col:\n" << patches_im2col << "\npatches_im2col dims: "
            << patches_im2col.dimensions() << "\n\n";

    Tensor<2> kernels_im2col = kernel.reshape(
            Dims<2>(filters, filter_size * filter_size * channels));
    std::cout << "kernels_im2col:\n" << kernels_im2col << "\nkernels_im2col dims: "
            << kernels_im2col.dimensions() << "\n\n";


    Tensor<3> convolution_contraction = patches_im2col.contract(kernels_im2col,
                                                                ContractDim{
                                                                        Axes(2, 1)});
    Tensor<4> convolution_reshaped = convolution_contraction.reshape(
            Dims<4>(batch, output_w_h, output_w_h, filters));

    // this one-liner is equivalent to all the above
    // uses the im2col method and generalizes for any number of batches, filters, and channels
    // inspiration from https://stackoverflow.com/questions/55532819/replicating-tensorflows-conv2d-operating-using-eigen-tensors
    // and the tensorflow file eigen_spatial_convolutions.h
    Tensor<4> conv = image
            .extract_image_patches(filter_size, filter_size, stride, stride, stride, stride, Eigen::PADDING_VALID)
            .reshape(Dims<3>(batch, patch_count, filter_size * filter_size * channels))
            .contract(kernel.reshape(Dims<2>(filters, filter_size * filter_size * channels)), ContractDim{Axes(2, 1)})
            .reshape(Dims<4>(batch, output_w_h, output_w_h, filters));

    std::cout << "one liner compact, convolution result:\n"
            << conv.shuffle(Dims<4>(0, 3, 1, 2))
            << "\nconvolution dims " << conv.dimensions() << "\n";

}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

//https://github.com/nlohmann/json
    //extract_img_patches_testing();
    im2col_convolution();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}