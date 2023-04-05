#include <iostream>
#include <chrono>
#include <flare/flare.hpp>
//#include "lib/json-develop/single_include/nlohmann/json.hpp"

void set_weights(fl::Tensor<2> &weights)
{
    using namespace fl;

    weights.slice(Dims<2>(0, 0), Dims<2>(3, 20)) = Tensor<2>(3, 20).setValues(
            {{0.576965, 0.659116, 0.702884, 0.192492, 0.783214, 0.92249,  0.337962, 0.194888, 0.298119, 0.238666, 0.272034, 0.358062, 0.019324, 0.213479, 0.590977, 0.988413, 0.609288, 0.084484, 0.810871,
                     0.044128},
             {0.53865,  0.951132, 0.430423, 0.663795, 0.346688, 0.433916, 0.298551, 0.112593, 0.396427, 0.022314, 0.699614, 0.981615, 0.723073, 0.091883, 0.719622, 0.122104, 0.493797, 0.257438, 0.799938,
                     0.889816},
             {0.633806, 0.98536,  0.091141, 0.950038, 0.247677, 0.349895, 0.789082, 0.997984, 0.162204, 0.039145, 0.613252, 0.344108, 0.400575, 0.89086,  0.456554, 0.49994,  0.340965, 0.695167, 0.358084,
                     0.268131}});
    weights.slice(Dims<2>(3, 0), Dims<2>(5, 20)) = Tensor<2>(5, 20).setValues(
            {{0.074563, 0.218356, 0.937712, 0.950692, 0.527635, 0.330107, 0.6655,   0.79536,  0.894386, 0.487504, 0.461727, 0.220733, 0.396802, 0.365143, 0.930384, 0.750919, 0.139293, 0.979788, 0.152928,
                     0.41032},
             {0.183174, 0.96891,  0.663228, 0.972345, 0.235832, 0.451922, 0.283095, 0.797042, 0.445114, 0.526298, 0.689257, 0.658285, 0.478264, 0.274353, 0.911132, 0.35455,  0.354401, 0.454006, 0.464945,
                     0.489994},
             {0.860942, 0.081264, 0.582303, 0.377804, 0.795148, 0.103755, 0.752929, 0.425141, 0.467988, 0.436143, 0.402033, 0.536081, 0.141227, 0.869395, 0.55777,  0.069341, 0.377727, 0.081909, 0.369297,
                     0.292048},
             {0.351103, 0.330927, 0.363646, 0.045541, 0.739394, 0.976308, 0.607712, 0.58135,  0.381556, 0.109799, 0.819477, 0.859921, 0.667143, 0.893585, 0.435746, 0.44293,  0.44404,  0.873219, 0.5029,
                     0.572638},
             {0.54924,  0.767787, 0.888377, 0.288708, 0.832866, 0.028553, 0.284296, 0.497932, 0.273681, 0.66692,  0.400512, 0.886717, 0.360105, 0.312042, 0.230181, 0.315914, 0.371168, 0.460909, 0.358639,
                     0.386638}});
}


void test()
{
    using namespace fl;

    Tensor<3> inputs(1, 2, 3);
    inputs.setConstant(1.0);
    inputs.setValues({{{0.0492, 0.423, 0.0654}, {0.312, 0.0534, 0.143}}});

    Tensor<2> weights(8, 20);
    set_weights(weights);

    Layer *bi_lstm = new Bidirectional<CONCAT, Sigmoid, TanH, true>(
            new LSTM<Sigmoid, TanH, true>(3, 5));
    bi_lstm->SetWeights(std::vector<Tensor<2>> {weights, {}, 2 * weights, {}});

    // forward propagation
    bi_lstm->Forward(inputs);
    std::cout << "output: " << bi_lstm->GetOutput3D().dimensions()
              << "\n" << bi_lstm->GetOutput3D() << "\n";

    // hard code a label tensor of ones
    MeanSquaredError<3> loss;
    loss(bi_lstm->GetOutput3D(), bi_lstm->GetOutput3D().constant(1.0));
    std::cout << "loss gradients: " << loss.GetGradients().dimensions()
              << "\n" << loss.GetGradients() << "\n";

    // backpropagation weights
    bi_lstm->Backward(loss.GetGradients());
    std::cout << bi_lstm->name << " weight gradients\n";
    for (auto &i: bi_lstm->GetWeightGradients2D()) {
        std::cout << i << "\n\n";
    }

    // backpropagation inputs
    std::cout << "input gradients\n" << bi_lstm->GetInputGradients3D() << "\n";
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