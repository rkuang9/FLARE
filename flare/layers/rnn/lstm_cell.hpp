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
    LSTMCell(Eigen::Index time_step, Eigen::Index input_len, Eigen::Index output_len)
            : time_step(time_step),
              input_len(input_len),
              output_len(output_len)
    {
        std::cout << "initialize cell: " << time_step << ", " << input_len << ", "
                  << output_len << "\n";
    }


    template<typename Device>
    void Forward(const Tensor<3> &x, const Tensor<2> &w,
                 Tensor<3> &h, Tensor<3> &cs,
                 const Device &device = Eigen::DefaultDevice())
    {
        // resize for multi-threading
        this->gates.resize(x.dimension(0), this->output_len * 4);
        this->p_gates.resize(this->gates.dimensions());
        this->x_h_prev.resize(x.dimension(0), this->input_len + this->output_len);

        this->x_h_prev
                .slice(Dims<2>(0, 0),
                       Dims<2>(x.dimension(0), this->input_len))
                .device(device) = x.chip(this->time_step, 1);

        if (this->time_step != 0) {
            this->x_h_prev
                    .slice(Dims<2>(0, this->input_len),
                           Dims<2>(x.dimension(0), this->output_len))
                    .device(device) = h.chip(this->time_step - 1, 1);
        }
        else {
            this->x_h_prev
                    .slice(Dims<2>(0, this->input_len),
                           Dims<2>(x.dimension(0), this->output_len))
                    .setZero();
        }

        this->p_gates = this->x_h_prev.contract(w, ContractDim {Axes(1, 0)});

        this->gates.slice(this->i_offset(), this->gate_extent()).device(device) =
                GateActivation::Activate(
                        this->p_gates.slice(this->i_offset(), this->gate_extent()));

        this->gates.slice(this->f_offset(), this->gate_extent()).device(device) =
                GateActivation::Activate(
                        this->p_gates.slice(this->f_offset(), this->gate_extent()));

        this->gates.slice(this->c_offset(), this->gate_extent()).device(device) =
                RecurrentActivation::Activate(
                        this->p_gates.slice(this->c_offset(), this->gate_extent()));

        this->gates.slice(this->o_offset(), this->gate_extent()).device(device) =
                GateActivation::Activate(
                        this->p_gates.slice(this->o_offset(), this->gate_extent()));

        cs.chip(this->time_step, 1).device(device) =
                this->gates.slice(this->f_offset(), this->gate_extent()) *
                (this->time_step == 0 ? Tensor<2>(1, 5).setZero() : cs.chip(
                        this->time_step - 1, 1)) +
                this->gates.slice(this->i_offset(), this->gate_extent()) *
                this->gates.slice(this->c_offset(), this->gate_extent());

        h.chip(this->time_step, 1).device(device) =
                this->gates.slice(this->o_offset(), this->gate_extent()) *
                RecurrentActivation::Activate(cs.chip(this->time_step, 1));
    }


    template<typename Device>
    void Backward(const Tensor<3> &gradients,
                  const Tensor<3> &x,
                  const Tensor<2> &w, Tensor<2> &dL_dw,
                  const Tensor<3> &h, const Tensor<3> &cs,
                  const Tensor<2> &dL_dh_next, const Tensor<2> &dL_dcs_next,
                  const Device &device = Eigen::DefaultDevice())
    {
        Tensor<2> dL_dcs(x.dimension(0), this->output_len);
        dL_dcs.device(device) =
                (gradients + dL_dh_next) *
                this->gates.slice(this->o_offset(), this->gate_extent()) *
                RecurrentActivation::Gradients(cs.chip(this->time_step, 1)) +
                dL_dcs_next;

        auto d_i = dL_dcs * this->gates.slice(this->c_offset(), this->gate_extent());
        auto d_f = dL_dcs * this->gates.slice(this->o_offset(), this->gate_extent());
        auto d_c = dL_dcs * this->gates.slice(this->i_offset(), this->gate_extent());
        auto d_o = (gradients + dL_dh_next) *
                   RecurrentActivation::Activate(cs.chip(this->time_step, 1));

        // calculate loss gradient w.r.t. pre-activation gates
        this->dp_gates.slice(this->i_offset(), this->gate_extent()).device(device) =
                d_i * GateActivation::Gradients(
                        this->p_gates.slice(this->i_offset(), this->gate_extent()));

        this->dp_gates.slice(this->f_offset(), this->gate_extent()).device(device) =
                d_f * GateActivation::Gradients(
                        this->p_gates.slice(this->f_offset(), this->gate_extent()));

        this->dp_gates.slice(this->c_offset(), this->gate_extent()).device(device) =
                d_c * GateActivation::Gradients(
                        this->p_gates.slice(this->c_offset(), this->gate_extent()));

        this->dp_gates.slice(this->o_offset(), this->gate_extent()).device(device) =
                d_o * GateActivation::Gradients(
                        this->p_gates.slice(this->o_offset(), this->gate_extent()));

        ContractDim weight_grad_matmul {Axes(0, 0)};

        dL_dw.slice(this->wu_i_offset(), this->wu_extent()).device(device) =
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->i_offset(), this->gate_extent()),
                        weight_grad_matmul);
        dL_dw.slice(this->wu_f_offset(), this->wu_extent()).device(device) =
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->f_offset(), this->gate_extent()),
                        weight_grad_matmul);
        dL_dw.slice(this->wu_c_offset(), this->wu_extent()).device(device) =
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->c_offset(), this->gate_extent()),
                        weight_grad_matmul);
        dL_dw.slice(this->wu_o_offset(), this->wu_extent()).device(device) =
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->o_offset(), this->gate_extent()),
                        weight_grad_matmul);

        if (this->time_step > 0) {
            // if not the first cell, calculate the input gradients for "h" and "cs"
            this->dh_prev.resize();
            this->dc_prev.resize();

            ContractDim h_grad_matmul {Axes(1, 1)};
            this->dh_prev.device(device) =
                    this->dp_gates.slice(this->i_offset(), this->gate_extent())
                            .contract(w.slice(this->ui_offset(), this->u_extent()),
                                      h_grad_matmul) +
                    this->dp_gates.slice(this->f_offset(), this->gate_extent())
                            .contract(w.slice(this->uf_offset(), this->u_extent()),
                                      h_grad_matmul) +
                    this->dp_gates.slice(this->c_offset(), this->gate_extent())
                            .contract(w.slice(this->uc_offset(), this->u_extent()),
                                      h_grad_matmul) +
                    this->dp_gates.slice(this->o_offset(), this->gate_extent())
                            .contract(w.slice(this->uo_offset(), this->u_extent()),
                                      h_grad_matmul);

            this->dcs_prev.device(device) = dL_dcs * this->gates.slice(
                    this->f_offset(), this->gate_extent());
        }
    }


    const Tensor<2> &GetInputGradientsHprev() const
    {
        return this->dL_dh_prev;
    }


    const Tensor<2> &GetInputGradientsCprev() const
    {
        return this->dL_dc_prev;
    }


    template<typename Device>
    void CalcInputGradients(Tensor<3> &dL_dx, const Tensor<2> &w,
                            const Device &device = Eigen::DefaultDevice())
    {
        ContractDim matmul {Axes(1, 1)};
        dL_dx.chip(this->time_step, 1).device(device) =
                this->dp_gates.slice(this->i_offset(), this->gate_extent())
                        .contract(w.slice(this->wi_offset(),
                                          this->w_extent()), matmul) +
                this->dp_gates.slice(this->f_offset(), this->gate_extent())
                        .contract(w.slice(this->wf_offset(),
                                          this->w_extent()), matmul) +
                this->dp_gates.slice(this->c_offset(), this->gate_extent())
                        .contract(w.slice(this->wc_offset(),
                                          this->w_extent()), matmul) +
                this->dp_gates.slice(this->o_offset(), this->gate_extent())
                        .contract(w.slice(this->wo_offset(),
                                          this->w_extent()), matmul);
    }


protected:
    inline Dims<2> i_offset()
    { return Dims<2>(0, 0); }


    inline Dims<2> f_offset()
    { return Dims<2>(0, this->output_len); }


    inline Dims<2> c_offset()
    { return Dims<2>(0, this->output_len * 2); }


    inline Dims<2> o_offset()
    { return Dims<2>(0, this->output_len * 3); }


    inline Dims<2> gate_extent()
    { return Dims<2>(this->gates.dimension(0), this->output_len); }


    // W:U weights offsets, extents
    inline Dims<2> wu_i_offset()
    { return Dims<2>(0, 0); }


    inline Dims<2> wu_f_offset()
    { return Dims<2>(0, this->output_len); }


    inline Dims<2> wu_c_offset()
    { return Dims<2>(0, this->output_len * 2); }


    inline Dims<2> wu_o_offset()
    { return Dims<2>(0, this->output_len * 3); }


    inline Dims<2> wu_extent()
    { return Dims<2>(this->input_len + this->output_len, this->output_len); }


    // W weights offsets, extents
    inline Dims<2> wi_offset()
    { return Dims<2>(0, 0); }


    inline Dims<2> wf_offset()
    { return Dims<2>(0, this->output_len * 1); }


    inline Dims<2> wc_offset()
    { return Dims<2>(0, this->output_len * 2); }


    inline Dims<2> wo_offset()
    { return Dims<2>(0, this->output_len * 3); }


    inline Dims<2> w_extent()
    { return Dims<2>(this->input_len, this->output_len); }


    // U weights offsets, extents
    inline Dims<2> ui_offset()
    { return Dims<2>(this->input_len, 0); }


    inline Dims<2> uf_offset()
    { return Dims<2>(this->input_len, this->output_len); }


    inline Dims<2> uc_offset()
    { return Dims<2>(this->input_len, this->output_len * 2); }


    inline Dims<2> uo_offset()
    { return Dims<2>(this->input_len, this->output_len * 3); }


    inline Dims<2> u_extent()
    { return Dims<2>(this->output_len, this->output_len); }


protected:
    const Eigen::Index time_step;
    const Eigen::Index input_len;
    const Eigen::Index output_len;

    Tensor<2> x_h_prev;

    Tensor<2> gates;
    Tensor<2> p_gates; // gates before activation
    Tensor<2> dp_gates; // loss gradients w.r.t. pre-activation gates

    Tensor<2> dh_prev;
    Tensor<2> dc_prev;

};

}

#endif //FLARE_LSTM_CELL_HPP
