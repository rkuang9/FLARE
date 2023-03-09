//
// Created by R on 2/20/23.
//

#ifndef ORION_GRU_CELL_HPP
#define ORION_GRU_CELL_HPP

#include "orion/orion_types.hpp"
#include "orion/orion_assert.hpp"

namespace orion
{

template<typename CandidateActivation = TanH, typename GateActivation = Sigmoid>
class GRUCell
{
public:
    explicit GRUCell(Eigen::Index time_step, Eigen::Index batch_size,
                     Eigen::Index input_length, Eigen::Index output_length)
            : time_step(time_step),
              batch_size(batch_size),
              input_feature_len(input_length),
              output_len(output_length)
    {
    }


    // calculate the GRU layer output at this cell's time step
    void Forward(const Tensor<3> &inputs, Tensor<3> &h, const Tensor<2> &w_zr,
                 const Tensor<2> &w_h, const Eigen::ThreadPoolDevice &device)
    {
        orion_assert(
                inputs.dimension(2) + h.dimension(2) == w_zr.dimension(0),
                "GRUCell" << this->time_step << "::Forward expected "
                          << w_zr.dimension(0)
                          << " features, got combined x_t and h_t_prev feature length "
                          << inputs.dimension(2) + h.dimension(2));

        orion_assert(w_zr.dimension(1) == 2.0 * this->output_len &&
                     w_h.dimension(1) == this->output_len,
                     "GRUCell" << this->time_step
                               << "::Forward received incorrect weights");
        // resize tensors to enable multi-threading
        this->pcandidate.resize(this->batch_size, 1, this->output_len);
        this->candidate.resize(this->batch_size, 1, this->output_len);
        this->pzr_gate.resize(this->batch_size, 1,
                              static_cast<Eigen::Index>(2) * this->output_len);
        this->zr_gate.resize(this->batch_size, 1,
                             static_cast<Eigen::Index>(2) * this->output_len);
        this->h_prev.resize(this->batch_size, 1, this->output_len);

        ContractDim contract_axes {Axes(2, 0)};

        // get the previous cell's output, or a tensor of zeroes if it's the first cell
        this->h_prev.device(device) =
                this->time_step == 0
                ? Tensor<3>(this->h_extent()).setZero()
                : h.slice(this->h_prev_offset(), this->h_extent());

        // calculate update and reset gates
        // since update and reset weights are come side-by-side as w_z:w_r,
        // concatenate x_t:h_prev to calculate gate values together
        // z:r = g(x_t x Wzr + h_prev x Uzr) = g(x_t:h_prev x Wzr)
        this->pzr_gate.device(device) =
                inputs.slice(this->x_offset(), this->x_extent())
                        .concatenate(this->h_prev, 2)
                        .contract(w_zr, contract_axes);
        this->zr_gate.device(device) = GateActivation::Activate(this->pzr_gate);

        // set up gate expressions
        auto update_gate = this->zr_gate.slice(this->z_offset(), this->z_extent());
        auto reset_gate = this->zr_gate.slice(this->r_offset(), this->r_extent());

        // calculate candidate output = g(x_t x W_h + (r_t *h_prev)xU_c)
        this->pcandidate.device(device) =
                inputs.slice(this->x_offset(), this->x_extent())
                        .concatenate(reset_gate * this->h_prev, 2)
                        .contract(w_h, contract_axes);
        this->candidate.template device(device) =
                CandidateActivation::Activate(this->pcandidate);

        // output: h_t = z * h_prev + (1 - z) * candidate
        // optimized equivalent: h_t = z * (h_prev - candidate) + candidate
        h.slice(this->h_offset(), this->h_extent()).device(device) =
                update_gate * (this->h_prev - this->candidate) + this->candidate;
    }


    // calculate the GRU layer weight gradients for this cell's time step
    void Backward(GRUCell &next_cell, const Tensor<3> &inputs,
                  const Tensor<2> &w_zr, Tensor<2> &dL_dw_zr,
                  const Tensor<2> &w_c, Tensor<2> &dL_dw_c,
                  const Tensor<3> &h, Tensor<3> &dL_dx,
                  const Eigen::ThreadPoolDevice &device)
    {
        this->Backward(next_cell.GetCellInputGradients(), inputs,
                       w_zr, dL_dw_zr,
                       w_c, dL_dw_c,
                       h, dL_dx, device);
    }


    // calculate the GRU layer weight gradients for this cell's time step
    void Backward(const Tensor<3> &gradients, const Tensor<3> &inputs,
                  const Tensor<2> &w_zr, Tensor<2> &dL_dw_zr,
                  const Tensor<2> &w_c, Tensor<2> &dL_dw_c,
                  const Tensor<3> &h, const Eigen::ThreadPoolDevice &device)
    {
        // resize pre-activation update, reset, and candidate gradients
        this->dL_dpzr.resize(this->pzr_gate.dimensions());
        this->dL_dpcand.resize(this->pcandidate.dimensions());

        // gate values after activation
        auto update_gate = this->zr_gate.slice(this->z_offset(), this->z_extent());
        auto reset_gate = this->zr_gate.slice(this->r_offset(), this->r_extent());

        // candidate and pre-activation candidate gradients
        // dL/dc = dL/dh * dh/dc
        // dL/dpc = dL/dc * dc/dpc = dL/dh * dh/dc * dc/dpc = dL/dc * g'(pc)
        this->dL_dpcand.device(device) =
                gradients * (1 - update_gate) *
                CandidateActivation::Gradients(this->pcandidate);

        // update gradients dL/dz = dL/dh * dh/dz = dL/dh * (h_prev - candidate)
        auto dL_dz = gradients * (this->h_prev - this->candidate);

        // reset gradients dL/dr = dL/dpcand * dpcand/dr = (dL/dcand x U_cand) * h_prev
        auto dL_dr = this->dL_dpcand.contract(
                w_c.slice(this->u_z_offset(), this->u_extent()),
                ContractDim {Axes {2, 1}}) * this->h_prev;

        // calculate pre-activation update and reset gate gradients
        // combine the two formulas
        // dL/dpz = dL/dz * dz/dz_ = dL/dz * g'(pz)
        // dL/dpr = dL/dr * dr/dr_ = dL/dr * g'(pr)
        // into the concatenated version dL/dpzr = dL/dz:dL/dr * g'(pzr)
        this->dL_dpzr.device(device) = dL_dz.concatenate(dL_dr, 2) *
                                       GateActivation::Gradients(this->pzr_gate);

        ContractDim weight_grad_axes {Axes(0, 0)};

        // the update:reset weights are calculated together
        // dL/dw_z = dz_/dw_z * dL/dz_ concat dL/dw_r = dr_/dw_r * dL/dr_
        dL_dw_zr.device(device) +=
                inputs.slice(this->x_offset(), this->x_extent())
                        .concatenate(this->h_prev, 2)
                        .contract(this->dL_dpzr, weight_grad_axes)
                        .reshape(w_zr.dimensions());

        // dL/dw_c = dc/dw_c * dL/dc
        dL_dw_c.device(device) += inputs.slice(x_offset(), x_extent())
                .concatenate(this->h_prev * reset_gate, 2)
                .contract(this->dL_dpcand, weight_grad_axes)
                .reshape(w_c.dimensions());

        // calculate the gradients for the previous cell's "h" output
        if (this->time_step != 0) {
            this->dL_dh_prev.resize(this->h_prev.dimensions());
            auto u_z = w_zr.slice(this->u_z_offset(), this->u_extent());
            auto u_r = w_zr.slice(this->u_r_offset(), this->u_extent());
            auto u_c = w_c.slice(this->u_z_offset(), this->u_extent());

            auto dL_dpz = this->dL_dpzr.slice(this->z_offset(), this->z_extent());
            auto dL_dpr = this->dL_dpzr.slice(this->r_offset(), this->r_extent());

            ContractDim matmul {Axes(2, 1)};

            this->dL_dh_prev.device(device) =
                    gradients * update_gate +
                    this->dL_dpcand.contract(u_c, matmul) * reset_gate +
                    dL_dpz.contract(u_z, matmul) +
                    dL_dpr.contract(u_r, matmul);
        }
    }


    const Tensor<3> &GetCellInputGradients()
    {
        return this->dL_dh_prev;
    }


    // calculate the GRU layer's input gradients at this cell's time step
    void CalcLayerInputGradients(
            const Tensor<2> &w_zr, const Tensor<2> &w_c, Tensor<3> &dL_dx,
            const Eigen::ThreadPoolDevice &device)
    {
        // dL/dx = dL/dpcand x w_c + dL/dpr x w_r + dL/dpz x w_z
        auto w_z = w_zr.slice(this->w_z_offset(), this->w_extent());
        auto w_r = w_zr.slice(this->w_r_offset(), this->w_extent());
        auto w_cand = w_c.slice(this->w_z_offset(), this->w_extent());
        auto dL_dpz = this->dL_dpzr.slice(this->z_offset(), this->z_extent());
        auto dL_dpr = this->dL_dpzr.slice(this->r_offset(), this->r_extent());

        ContractDim matmul {Axes(2, 1)};

        dL_dx.slice(this->x_offset(), this->x_extent()).device(device) =
                dL_dpz.contract(w_z, matmul) + dL_dpr.contract(w_r, matmul) +
                this->dL_dpcand.contract(w_cand, matmul);
    }


protected:
    inline Dims<3> x_offset() const
    { return Dims<3> {0, this->time_step, 0}; }


    inline Dims<3> x_extent() const
    { return Dims<3>(this->batch_size, 1, this->input_feature_len); }


    inline Dims<3> h_offset() const
    { return Dims<3>(0, this->time_step, 0); }


    inline Dims<3> h_prev_offset() const
    { return Dims<3>(0, this->time_step - 1, 0); }


    inline Dims<3> h_extent() const
    { return Dims<3>(this->batch_size, 1, this->output_len); }


    inline Dims<3> z_offset() const
    { return Dims<3>(0, 0, 0); }


    inline Dims<3> z_extent() const
    { return Dims<3>(this->batch_size, 1, this->output_len); }


    inline Dims<3> r_offset() const
    { return Dims<3>(0, 0, this->output_len); }


    inline Dims<3> r_extent() const
    { return Dims<3>(this->batch_size, 1, this->output_len); }


    inline Dims<2> w_z_offset() const
    { return Dims<2>(0, 0); }


    inline Dims<2> w_r_offset() const
    { return Dims<2>(0, this->output_len); }


    inline Dims<2> w_extent() const
    { return Dims<2>(this->input_feature_len, this->output_len); }


    inline Dims<2> u_z_offset() const
    { return Dims<2>(this->input_feature_len, 0); }


    inline Dims<2> u_r_offset() const
    { return Dims<2>(this->input_feature_len, this->output_len); }


    inline Dims<2> u_extent() const
    { return Dims<2>(this->output_len, this->output_len); }


private:
    Tensor<3> zr_gate; // update and reset gate stacked side-by-side
    Tensor<3> pzr_gate; // pre-activation update and reset gate

    Tensor<3> candidate; // candidate output
    Tensor<3> pcandidate; // pre-activation candidate output

    Tensor<3> h_prev;
    //Tensor<3> h; // declared in GRU layer instead of GRUCell

    // loss gradients w.r.t. update and reset gate, candidate, and prev cell output
    Tensor<3> dL_dpzr;
    Tensor<3> dL_dpcand;
    Tensor<3> dL_dh_prev;

    Eigen::Index batch_size;
    const Eigen::Index input_feature_len;
    const Eigen::Index output_len;
    const Eigen::Index time_step;

};

} // namespace orion

#endif //ORION_GRU_CELL_HPP
