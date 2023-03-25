#include <iostream>
#include <chrono>
#include <flare/flare.hpp>


void test()
{
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(
            new Eigen::ThreadPool(8), 2);

    fl::MeanSquaredError<3> loss;

    Eigen::Index batch_size = 2;
    Eigen::Index input_len = 3;
    Eigen::Index output_len = 5;
    Eigen::Index time_sequence = 3;

    fl::Tensor<2> w(input_len + output_len, output_len * 4);
    fl::Tensor<2> dL_dw(w.dimensions());
    w.setConstant(1.0);
    dL_dw.setZero();

    fl::Tensor<3> x(batch_size, time_sequence, input_len);
    x.setConstant(1.0);
    fl::Tensor<3> h(batch_size, time_sequence, output_len);
    fl::Tensor<3> cs(batch_size, time_sequence, output_len);
    fl::Tensor<3> y(batch_size, time_sequence, output_len);
    y.setConstant(1.0);

    std::vector<fl::LSTMCell<fl::Linear, fl::Linear>> cells;
    cells.reserve(time_sequence);

    for (int i = 0; i < time_sequence; i++) {
        cells.emplace_back(i, input_len, output_len);
    }

    for (int i = 0; i < time_sequence; i++) {
        cells[i].Forward(x, w, h, cs, device);
    }

    loss(h, y);
    std::cout << "loss gradients: " << loss.GetGradients().dimensions() << "\n" << loss.GetGradients() << "\n";

    std::cout << "h output: " << h.dimensions() << "\n" << h << "\n";

    cells.back().Backward(
            loss.GetGradients(),
            w, dL_dw,
            h, cs,
            fl::Tensor<2>(batch_size, output_len).setZero(),
            fl::Tensor<2>(batch_size, output_len).setZero(),
            device
    );

    for (auto i = time_sequence - 2; i >= 0; i--) {
        std::cout << "run cell " << i << " backward\n";
        cells[i].Backward(
                loss.GetGradients(),
                w, dL_dw,
                h, cs,
                cells[i + 1].GetInputGradientsHprev(),
                cells[i + 1].GetInputGradientsCprev(),
                device
        );
    }

    std::cout << "dL/dw: " << dL_dw.dimensions() << "\n" << dL_dw << "\n";
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