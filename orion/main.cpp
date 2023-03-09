#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


using namespace orion;


void test()
{
    Tensor<3> inputs(2, 3, 3);
    inputs.setValues({{{0.75674543, 0.48890536, 0.3872954},
                              {0.99806318, 0.25933461, 0.83642692},
                              {0.83294866, 0.78341476, 0.34627345}},
                      {{0.12554334, 0.08473995, 0.40768565},
                              {0.71470978, 0.86196027, 0.09974493},
                              {0.14966181, 0.92493361, 0.13505977}
                      }});

    Sequential model {
            new GRU<TanH, Sigmoid>(3, 5),
    };

    MeanSquaredError<3> mse;
    SGD opt(1.0);

    model.Forward(inputs);
    std::cout << "output: " << model.layers.back()->GetOutput3D().dimensions()
              << "\n" << model.layers.back()->GetOutput3D() << "\n";

    mse(model.layers.back()->GetOutput3D(),
        model.layers.back()->GetOutput3D().constant(1.0));
    std::cout << "loss gradients\n" << mse.GetGradients() << "\n";
    model.Backward(mse.GetGradients());
    std::cout << "input gradients dL/dx\n"
              << model.layers.back()->GetInputGradients3D() << "\n";
    model.Update(opt);

    model.Forward(inputs);
    std::cout << "output: " << model.layers.back()->GetOutput3D().dimensions()
              << "\n" << model.layers.back()->GetOutput3D() << "\n";

    mse(model.layers.back()->GetOutput3D(),
        model.layers.back()->GetOutput3D().constant(1.0));
    std::cout << "loss gradients\n" << mse.GetGradients() << "\n";
    model.Backward(mse.GetGradients());
    std::cout << "input gradients dL/dx\n"
              << model.layers.back()->GetInputGradients3D() << "\n";
    model.Update(opt);

}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    test();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic