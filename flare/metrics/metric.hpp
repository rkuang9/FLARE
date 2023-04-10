//
// Created by R on 4/9/23.
//

#ifndef FLARE_METRIC_HPP
#define FLARE_METRIC_HPP

#include "flare/fl_types.hpp"
#include "flare/fl_assert.hpp"

namespace fl
{

template<int TensorRank>
class Metric
{
public:
    virtual void operator()(const Tensor <TensorRank> &label,
                            const Tensor <TensorRank> &pred) = 0;

    virtual double GetMetric() const = 0;

    virtual void Reset() = 0;


    friend std::ostream &operator<<(std::ostream &out,
                                    const Metric<TensorRank> &metric)
    {
        return out << metric.name << ": " << metric.GetMetric();
    }

    std::string name = "metric";

protected:
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(new Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency()), 2);
};

} // namespace fl

#endif //FLARE_METRIC_HPP
