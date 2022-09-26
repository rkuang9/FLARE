//
// Created by RKuang on 9/25/2022.
//

#include "loss.hpp"

namespace orion
{
    Loss::Loss()
    {
        this->name = "loss";
    }


    Scalar Loss::Compute(const Sequential &model) const
    {
        const LossFunction *loss = model.GetLossFunction();

        // total loss from all mini-batches of the latest epoch
        auto running_loss = static_cast<Scalar>(0.0);

        // indices on which to sum loss values
        int start = loss->LossHistory().size() - model.GetTotalSamples();
        int end = loss->LossHistory().size();

        for (int i = start; i < end; i++) {
            running_loss += loss->LossHistory()[i];
        }

        return running_loss / model.GetTotalSamples();
    }
}