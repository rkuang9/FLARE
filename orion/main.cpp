#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


void Conv2DTransposeSame()
{
    using namespace orion;
    int filters = 1;
    int batch = 1;
    int channels = 1;
    Tensor<2> _input(5, 5);
    _input.setValues({{1, 1, 0, 1, 1},
                      {2, 2, 0, 2, 2},
                      {1, 2, 3, 4, 5},
                      {3, 3, 0, 3, 3},
                      {1, 1, 0, 1, 1}});
    Tensor<4> input = _input
            .reshape(Dims<4>(1, _input.dimension(0), _input.dimension(1), 1))
            .broadcast(Dims<4>(batch, 1, 1, channels));

    MeanSquaredError loss;
    SGD opt(1.0);

    Tensor<4> _kernels(5, 5, 1, 1);
    _kernels.setValues({{{{-0.3430285984}},
                                {{-0.0219324651}},
                                {{-0.1391617156}},
                                {{0.1771631229}},
                                {{0.1127058849}}},
                        {{{-0.1082837004}},
                                {{0.2101224378}},
                                {{0.0805285135}},
                                {{0.1240991481}},
                                {{-0.0909266167}}},
                        {{{0.312399834}},
                                {{0.1930091693}},
                                {{-0.3332852664}},
                                {{-0.0373753809}},
                                {{-0.1927529123}}},
                        {{{0.0727412922}},
                                {{0.3408326764}},
                                {{-0.0803240835}},
                                {{0.0142777371}},
                                {{-0.2310841138}}},
                        {{{-0.3392456519}},
                                {{0.327178136}},
                                {{0.1408941963}},
                                {{-0.2029289165}},
                                {{-0.0010828212}}}});

    Tensor<4> kernels = _kernels.shuffle(Dims<4>(3, 0, 1, 2));

    Layer *upscale = new Conv2DTranspose<Linear>(
            1, Input(5, 5, 1), Kernel(kernels.dimension(1), kernels.dimension(2)),
            Stride(2, 2), Dilation(1, 1), Padding::PADDING_VALID, Dims<2>(0, 0));
    upscale->SetWeights(kernels);
    upscale->Forward(input);
    std::cout << "output: " << upscale->GetOutput4D().dimensions() << "\n"
              << upscale->GetOutput4D().shuffle(NCHW) << "\n";

    loss.CalculateLoss(upscale->GetOutput4D(), upscale->GetOutput4D().constant(0));
    upscale->Backward(loss);

    std::cout << "weight grads\n" << upscale->GetWeightGradients4D().dimensions()
              << "\n" << upscale->GetWeightGradients4D().shuffle(NCHW) << "\n";

    std::cout << "input grads\n" << upscale->GetInputGradients4D().dimensions()
              << "\n" << upscale->GetInputGradients4D().shuffle(NCHW) << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    Conv2DTransposeSame();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic