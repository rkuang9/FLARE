#include <iostream>
#include <chrono>
#include <vector>
#include <orion/orion.hpp>
//#include "examples/xor_classifier.hpp"
#include "opencv2/core.hpp"


// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

using namespace orion;

static void help(char** argv)
{
    using namespace std;
    cout
            << "\n------------------------------------------------------------------\n"
            << " This program shows the serial out capabilities of cv::Mat\n"
            << "That is, cv::Mat M(...); cout << M;  Now works.\n"
            << "Output can be formatted to OpenCV, matlab, python, numpy, csv and \n"
            << "C styles Usage:\n"
            << argv[0]
            << "\n------------------------------------------------------------------\n\n"
            << endl;
}

void opencv()
{

}


void maxpool()
{
    int batches = 2;
    int channels = 3;
    int filters = 2;

    // hard code a small image
    Tensor<2> _fakeimg(3, 3);
    _fakeimg.setValues({{0.321, 0.542,  0.876},
                        {0.056, 0.0312, 0.432},
                        {0.432, 0.654,  0.192}});
    Tensor<4> fakeimg = _fakeimg
            .reshape(Dims<4>(1, 3, 3, 1))
            .broadcast(Dims<4>(batches, 1, 1, channels));

    // hard code the labels
    Tensor<4> fakelabels(batches, 2, 2, filters);
    fakelabels.setZero();

    std::vector<Tensor<4>> training_samples {fakeimg};
    std::vector<Tensor<4>> training_labels {fakelabels};

    Sequential model {
            new Conv2D<TanH>(filters, Input {3, 3, channels}, Kernel {2, 2},
                             Stride {1, 1}, Dilation {1, 1},
                             Padding::PADDING_VALID),
            new MaxPooling2D(PoolSize(3, 3), Stride(1, 1), Padding::PADDING_SAME),
    };

    MeanSquaredError loss;
    SGD opt(1);

    // hard code the kernel values
    Tensor<4> kernel(filters, 2, 2, channels);
    kernel.setConstant(1);
    model.layers[0]->SetWeights(kernel);

    model.Compile(loss, opt);
    model.Fit(training_samples, training_labels, 100, 1);

    std::cout << "loss: " << loss.GetLoss() << "\n";

    std::cout << "updated kernels:\n"
              << model.layers[0]->GetWeights4D().shuffle(Dims<4>(3, 0, 1, 2))
              << "\n";

    std::cout << "network output: \n"
              << model.Predict(fakeimg).shuffle(Dims<4>(3, 0, 1, 2)) << "\n";
}


int main(int argc, char** argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    opencv();

    using namespace std;
    using namespace std;
    using namespace cv;

    cv::CommandLineParser parser(argc, argv, "{help h||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    Mat I = Mat::eye(4, 4, CV_64F);
    I.at<double>(1,1) = CV_PI;
    cout << "I = \n" << I << ";" << endl << endl;
    Mat r = Mat(10, 3, CV_8UC3);
    randu(r, cv::Scalar::all(0), cv::Scalar::all(255));
    cout << "r (default) = \n" << r << ";" << endl << endl;
    cout << "r (matlab) = \n" << format(r, Formatter::FMT_MATLAB) << ";" << endl << endl;
    cout << "r (python) = \n" << format(r, Formatter::FMT_PYTHON) << ";" << endl << endl;
    cout << "r (numpy) = \n" << format(r, Formatter::FMT_NUMPY) << ";" << endl << endl;
    cout << "r (csv) = \n" << format(r, Formatter::FMT_CSV) << ";" << endl << endl;
    cout << "r (c) = \n" << format(r, Formatter::FMT_C) << ";" << endl << endl;
    Point2f p(5, 1);
    cout << "p = " << p << ";" << endl;
    Point3f p3f(2, 6, 7);
    cout << "p3f = " << p3f << ";" << endl;
    vector<float> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    cout << "shortvec = " << Mat(v) << endl;
    vector<Point2f> points(20);
    for (size_t i = 0; i < points.size(); ++i)
        points[i] = Point2f((float)(i * 5), (float)(i % 7));
    cout << "points = " << points << ";" << endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}