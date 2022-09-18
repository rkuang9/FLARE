//
// Created by RKuang on 9/17/2022.
//

#include "globalaveragepooling1d.hpp"

namespace orion {

GlobalAveragePooling1D::GlobalAveragePooling1D(Eigen::Index dimension) : target_dim(dimension) {
}

void GlobalAveragePooling1D::Forward(const Tensor<3> &inputs) {
    Layer::Forward(inputs);
}

void GlobalAveragePooling1D::Forward(const Layer &prev) {
    Layer::Forward(prev);
}

void GlobalAveragePooling1D::Backward(const Layer &next) {
    Layer::Backward(next);
}

const Tensor<2> &GlobalAveragePooling1D::GetOutput2D() const {
    return Layer::GetOutput2D();
}

const Tensor<3> &GlobalAveragePooling1D::GetOutput3D() const {
    return Layer::GetOutput3D();
}

const Tensor<2> &GlobalAveragePooling1D::GetInputGradients2D() const {
    return Layer::GetInputGradients2D();
}

int GlobalAveragePooling1D::GetInputRank() const {
    return 0;
}

int GlobalAveragePooling1D::GetOutputRank() const {
    return 0;
}

}