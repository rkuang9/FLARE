#include <iostream>
#include <chrono>
#include <flare/flare.hpp>


void test()
{
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(
            new Eigen::ThreadPool(8), 2);

    Eigen::Index batch_size = 1;
    Eigen::Index input_len = 3;
    Eigen::Index output_len = 5;
    Eigen::Index time_sequence = 5;

    fl::Tensor<2> w(input_len + output_len, output_len * 4);
    w.setConstant(1.0);

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

    std::cout << "h output: " << h.dimensions() << "\n" << h << "\n";

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