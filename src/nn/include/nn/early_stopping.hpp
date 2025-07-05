//
// Created by VanHoai on 13/6/25.
//

#ifndef EARLY_STOPPING_HPP
#define EARLY_STOPPING_HPP

#include <Eigen/Core>
#include <vector>

namespace nn {
enum class MonitorEarlyStopping {
    ValidationLoss,
    ValidationAccuracy,
};

class EarlyStopping {
private:
    int patience;
    double minDelta;
    bool isStore;
    MonitorEarlyStopping monitor;

    // Properties for tracking with training
    double bestScore;
    int wait;
    std::vector<Eigen::MatrixXd> Ws;
    std::vector<Eigen::MatrixXd> bs;
    int stoppedEpoch;

public:
    explicit EarlyStopping(
        const int patience = 10, const double minDelta = 1e-3, const bool isStore = false,
        const MonitorEarlyStopping monitor = MonitorEarlyStopping::ValidationLoss)
        : patience(patience), minDelta(minDelta), isStore(isStore), monitor(monitor) {
        this->wait = 0;
        this->stoppedEpoch = -1;
        if (monitor == MonitorEarlyStopping::ValidationLoss) {
            // Lower is better for loss
            this->bestScore = std::numeric_limits<double>::max();
        } else {
            // Higher is better for accuracy
            this->bestScore = -std::numeric_limits<double>::max();
        }

        if (isStore) {
            this->Ws = std::vector<Eigen::MatrixXd>();
            this->bs = std::vector<Eigen::MatrixXd>();
        }
    }

    [[nodiscard]] bool getIsStore() const { return isStore; }

    bool on_epoch_end(int epoch, double currentScore, const std::vector<Eigen::MatrixXd> &Ws,
                      const std::vector<Eigen::MatrixXd> &bs);
};

}  // namespace nn

#endif  // EARLY_STOPPING_HPP
