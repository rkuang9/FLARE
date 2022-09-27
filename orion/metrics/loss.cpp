//
// Created by RKuang on 9/25/2022.
//

#include "loss.hpp"
#include "orion/loss/loss_function.hpp"
#include "orion/sequential.hpp"

namespace orion
{

Loss::Loss()
{
    this->name = "loss";
}


Scalar Loss::Compute(Sequential &model) const
{
    const LossFunction *loss = model.GetLossFunction();

    int mini_batches = model.GetTotalSamples() / model.GetBatchSize();

    // total loss from all mini-batches of the latest epoch
    auto epoch_loss = static_cast<Scalar>(0.0);

    // indices of loss history on which to sum loss values
    int start = loss->LossHistory().size() - mini_batches;
    int end = loss->LossHistory().size();

    for (int i = start; i < end; i++) {
        epoch_loss += loss->LossHistory()[i];
    }

    // average the loss across current epoch's mini-batches
    return epoch_loss / mini_batches;
}

}