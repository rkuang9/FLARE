#include <iostream>
#include <chrono>
#include <vector>


#include <orion/orion.hpp>


using namespace orion;


auto SwapNHWCToNWHC(const Tensor<4> &tensor)
{
    Tensor<4>::Dimensions shuffle_hw = Tensor<4>::Dimensions(0, 2, 1, 3);
    // reverse the values in the col dim (3)
    Tensor<4>::Dimensions reverse_hw =
            Tensor<4, bool>::Dimensions(false, false, true, false);

    return tensor.shuffle(shuffle_hw).reverse(reverse_hw);
}


auto SwapNWHCToNHWC(const Tensor<4> &tensor)
{
    Tensor<4>::Dimensions shuffle_hw = Tensor<4>::Dimensions(0, 2, 1, 3);
    // reverse the values in the col dim (3)
    Tensor<4>::Dimensions reverse_hw =
            Tensor<4, bool>::Dimensions(false, true, false, false);

    return tensor.shuffle(shuffle_hw).reverse(reverse_hw);
}


/**
 * RowMajor expects NWHC (batch, width, height, channels)
 * Need data format NHWC (batch, height, width, channels)
 */

void im2col_convolution()
{
    int batch = 2;
    int channels = 10;
    int filters = 3;
    int filter_size = 2;
    int stride = 1;
    int output_w_h = 2;

    Tensor<3> preimage(1, 3, 3);
    preimage.setValues({{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}});
    Tensor<3> prekernel(1, 2, 2);
    prekernel.setValues({{{1, 1}, {2, 2}}});

    Tensor<4> image = preimage.reshape(Dims<4>(1, 3, 3, 1)).broadcast(Dims<4>(batch, 1, 1, channels));
    Tensor<5> kernel = prekernel.reshape(Dims<5>(1, 1, 1, filter_size, filter_size)).broadcast(Dims<5>(batch, filters, 1, 1, channels));

    std::cout << "kernel:\n" << kernel << "\nkernel dims: " << kernel.dimensions() << "\n\n";

    // for row-major, extract_image_patches returns (NPWHC), we shuffle it back to NPCWH where P = #patches
    Tensor<5> patches = image
            .extract_image_patches(filter_size, filter_size, stride, stride, stride, stride, Eigen::PADDING_VALID);
            //.shuffle(Dims<5>(0, 1, 4, 2, 3)); // don't mess with the dimensions yet
    std::cout << patches << "\npatches dims: " << patches.dimensions() << "\n\n";

    Eigen::Index patch_count = patches.dimension(1);


    Tensor<3> patches_im2col = patches.reshape(Dims<3>(batch, patch_count, filter_size * filter_size * channels));
    std::cout << "patches_im2col:\n" << patches_im2col << "\npatches_im2col dims: " << patches_im2col.dimensions() << "\n\n";

    Tensor<3> kernels_im2col = kernel.reshape(Dims<3>(batch, filters, filter_size * filter_size * channels));
    std::cout << "kernels_im2col:\n" << kernels_im2col << "\nkernels_im2col dims: " << kernels_im2col.dimensions() << "\n\n";


    Tensor<3> convolution_contraction = patches_im2col.contract(kernels_im2col.chip(0, 0), ContractDim{Axes(2, 1)});
    //std::cout << convolution_contraction << "\n\n";
    Tensor<4> convolution_reshaped = convolution_contraction.reshape(Dims<4>(batch, output_w_h, output_w_h, filters));
    //std::cout << convolution_reshaped << "\n\n";
    std::cout << convolution_reshaped.shuffle(Dims<4>(0, 3, 1, 2)) << "\npreshuffle convolution dims " << convolution_reshaped.dimensions() << "\n";
}


void img_batch_print()
{
    Dims<4> image_dims(2, 3, 3, 1);
    Tensor<4>::Dimensions kernel_dims(1, 2, 2, 1);
    Tensor<2>::Dimensions output_dims(2, 2);

    Tensor<2> input(3, 3);
    for (Eigen::Index i = 0; i < input.size(); i++) {
        input(i) = Scalar(i);
    }

    Tensor<2> kern(2, 2);
    for (Eigen::Index i = 0; i < kern.size(); i++) {
        kern(i) = Scalar(i) + Scalar(1);
    }

    Tensor<4> image = input.reshape(Dims<4>(1, 3, 3, 1)).broadcast(Dims<4>(2, 1, 1, 1));
    Tensor<4> kernel = kern.reshape(kernel_dims);

    PrintNHWCAsNCHW(image);
    PrintNHWCAsNCHW(kernel);

    Tensor<5> patches = image.
            extract_image_patches(kernel.dimension(1), kernel.dimension(2),
                                  1, 1, 1, 1, Eigen::PADDING_VALID);

    PrintPatches(patches);

    std::cout << patches.reshape(
            Tensor<2>::Dimensions(kern.size(), output_dims[0] * output_dims[1]))
            << "\n\n";

    std::cout << kernel.reshape(Tensor<2>::Dimensions(
            kernel.dimension(0) * kernel.dimension(1) * kernel.dimension(2),
            kernel.dimension(3))) << "\n\n";


    Tensor<2> output = image
            .extract_image_patches(kernel.dimension(1), kernel.dimension(2),
                                   1, 1, 1, 1, Eigen::PADDING_VALID)
            .reshape(Tensor<2>::Dimensions(
                    kern.size(), output_dims[0] * output_dims[1]))
            .contract(kernel.
                    reshape(Tensor<2>::Dimensions(
                    kernel.dimension(0) * kernel.dimension(1) * kernel.dimension(2),
                    kernel.dimension(3))), ContractDim{Axes(1, 0)});

    //std::cout << "convolved to\n" << output << "\n";
    std::cout << output.dimensions() << "\n\n";
    std::cout << output.reshape(Tensor<4>::Dimensions(
            image_dims[0], output_dims[0], output_dims[1], image_dims[3])).shuffle(Tensor<4>::Dimensions(0, 3, 1, 2));
            /*.reshape(Tensor<4>::Dimensions(
                    image_dims[0], output_dims[0], output_dims[1], image_dims[3]))
            .shuffle(Tensor<4>::Dimensions(0, 3, 1, 2)); // print as NCHW*/
}


// col-major: CHWN (channels, height, width, batch)
void extract_img_patches_testing()
{
    // 2 RGB images with size 5x5 pixels
    Eigen::Tensor<double, 4> img(3, 5, 5, 2);
    img.setRandom();
    /*for (int channels = 0; channels < img.dimension(0); channels++) {
        int index = 0;
        for (int row = 0; row < img.dimension(1); row++) {
            for (int col = 0; col < img.dimension(2); col++) {
                img(channels, row, col) = index;
                index++;
            }
        }
    }*/

    //std::cout << "input image\n" << img << "\n----------\n";

    Eigen::Tensor<double, 5> patches = img.extract_image_patches(
            3, 3, // kernel height/width
            1, 1, // stride height/width
            1, 1, // in_stride height/width
            Eigen::PADDING_VALID);


    std::cout << patches << "\npatch rank: " << patches.rank() << ", patch dims: "
            << patches.dimensions() << "\n";

    /*for (int i = 0; i < patches.dimension(3); i++) {
        std::cout << "patch " << i << "\n" << patches.chip(i, 3) << "\n";
        Tensor<4> dims = patches.chip(i, 3);
        std::cout << "chip dims: " << dims.dimensions() << "\n";
    }*/
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