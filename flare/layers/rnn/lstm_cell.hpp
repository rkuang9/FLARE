//
// Created by R on 3/11/23.
//

#include "flare/fl_types.hpp"
#include "flare/fl_assert.hpp"

#ifndef FLARE_LSTM_CELL_HPP
#define FLARE_LSTM_CELL_HPP

namespace fl
{

template<typename Activation = TanH, typename GateActivation = Sigmoid>
class LSTMCell
{
public:
    LSTMCell(Eigen::Index time_step, Eigen::Index input_len, Eigen::Index output_len)
            : time_step(time_step),
              input_len(input_len),
              output_len(output_len)
    {
        // nothing to do
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
        this->cs_prev.resize(x.dimension(0), this->output_len);

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
                Activation::Activate( // TODO: Gate Or Activation?
                        this->p_gates.slice(this->c_offset(), this->gate_extent()));

        this->gates.slice(this->o_offset(), this->gate_extent()).device(device) =
                GateActivation::Activate(
                        this->p_gates.slice(this->o_offset(), this->gate_extent()));

        this->cs_prev.device(device) = this->time_step == 0
                                       ? Tensor<2>(x.dimension(0),
                                                   this->output_len).setZero()
                                       : cs.chip(this->time_step - 1, 1);

        cs.chip(this->time_step, 1).device(device) =
                this->gates.slice(this->f_offset(), this->gate_extent()) *
                this->cs_prev +
                this->gates.slice(this->i_offset(), this->gate_extent()) *
                this->gates.slice(this->c_offset(), this->gate_extent());

        h.chip(this->time_step, 1).device(device) =
                this->gates.slice(this->o_offset(), this->gate_extent()) *
                Activation::Activate(cs.chip(this->time_step, 1));
    }


    template<typename Device>
    void Backward(const Tensor<2> &gradients,
                  const Tensor<2> &w, Tensor<2> &dL_dw,
                  const Tensor<3> &cs, const Tensor<2> &dcs_next,
                  const Device &device = Eigen::DefaultDevice())
    {
        this->dp_gates.resize(this->gates.dimensions());

        Tensor<2> dcs(cs.dimension(0), this->output_len);
        dcs.device(device) =
                gradients *
                this->gates.slice(this->o_offset(), this->gate_extent()) *
                Activation::Gradients(cs.chip(this->time_step, 1)) + // TODO: Gate or Activation?
                dcs_next;

        auto d_i = dcs * this->gates.slice(this->c_offset(), this->gate_extent());
        auto d_f = dcs * this->cs_prev;
        auto d_c = dcs * this->gates.slice(this->i_offset(), this->gate_extent());
        auto d_o = gradients *
                   Activation::Activate(cs.chip(this->time_step, 1));

        // calculate loss gradient w.r.t. pre-activation gates
        this->dp_gates.slice(this->i_offset(), this->gate_extent()).device(device) =
                d_i * GateActivation::Gradients(
                        this->p_gates.slice(this->i_offset(), this->gate_extent()));

        this->dp_gates.slice(this->f_offset(), this->gate_extent()).device(device) =
                d_f * GateActivation::Gradients(
                        this->p_gates.slice(this->f_offset(), this->gate_extent()));

        this->dp_gates.slice(this->c_offset(), this->gate_extent()).device(device) =
                d_c * Activation::Gradients(
                        this->p_gates.slice(this->c_offset(), this->gate_extent()));

        this->dp_gates.slice(this->o_offset(), this->gate_extent()).device(device) =
                d_o * GateActivation::Gradients(
                        this->p_gates.slice(this->o_offset(), this->gate_extent()));

        // calculate loss gradients w.r.t. weights
        ContractDim weight_grad_matmul {Axes(0, 0)};

        dL_dw.slice(this->wu_i_offset(), this->wu_extent()).device(device) +=
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->i_offset(), this->gate_extent()),
                        weight_grad_matmul);
        dL_dw.slice(this->wu_f_offset(), this->wu_extent()).device(device) +=
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->f_offset(), this->gate_extent()),
                        weight_grad_matmul);
        dL_dw.slice(this->wu_c_offset(), this->wu_extent()).device(device) +=
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->c_offset(), this->gate_extent()),
                        weight_grad_matmul);
        dL_dw.slice(this->wu_o_offset(), this->wu_extent()).device(device) +=
                this->x_h_prev.contract(
                        this->dp_gates.slice(this->o_offset(), this->gate_extent()),
                        weight_grad_matmul);

        if (this->time_step > 0) {
            // calculate loss gradients w.r.t. input "h" and "cs" from previous cell
            this->dh_prev.resize(gradients.dimensions());
            this->dcs_prev.resize(dcs_next.dimensions());

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

            this->dcs_prev.device(device) = dcs * this->gates.slice(
                    this->f_offset(), this->gate_extent());
        }
    }


    const Tensor<2> &GetInputGradientsHprev() const
    {
        return this->dh_prev;
    }


    const Tensor<2> &GetInputGradientsCprev() const
    {
        return this->dcs_prev;
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
    Tensor<2> dcs_prev;
    Tensor<2> cs_prev;

};

}

#endif //FLARE_LSTM_CELL_HPP
