//
// Created by R on 3/11/23.
//

#include "flare/fl_types.hpp"
#include "flare/fl_assert.hpp"

#ifndef FLARE_LSTM_CELL_HPP
#define FLARE_LSTM_CELL_HPP

namespace fl
{

template<typename GateActivation = Sigmoid, typename RecurrentActivation = TanH>
class LSTMCell
{
public:
    LSTMCell(Eigen::Index input_len, Eigen::Index output_len, Eigen::Index time_step)
            : time_step(time_step)
    {

    }


    template<typename Device>
    void Forward(const Tensor<3> &x, Tensor<3> &h, const Tensor<3> &c_prev,
                 const Device &device)
    {
        this->gates.resize(x.dimension(0), this->output_len * 4);

    }


protected:
    inline Dims<3> x_offset() const
    { return Dims<3>(0, this->time_step, 0); }


    inline Dims<3> x_extent() const
    { return Dims<3>(this->batch_size, 1, this->input_len); }


    inline Dims<3> h_prev_offset() const
    { return Dims<3>(0, this->time_step, 0); }


    inline Dims<3> h_extent() const
    { return Dims<3>(this->batch_size, 1, this->output_len); }


protected:
    const Eigen::Index time_step;
    Eigen::Index batch_size = -1; // set during forward propagation
    const Eigen::Index input_len;
    const Eigen::Index output_len;

    Tensor<3> gates;

};

}

#endif //FLARE_LSTM_CELL_HPP
