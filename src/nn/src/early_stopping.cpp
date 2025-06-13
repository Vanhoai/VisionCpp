//
// Created by VanHoai on 13/6/25.
//

#include "nn/early_stopping.hpp"

namespace nn {

    bool EarlyStopping::on_epoch_end(const int epoch, const double currentScore,
                                     const std::vector<Eigen::MatrixXd> &Ws,
                                     const std::vector<Eigen::MatrixXd> &bs) {
        if (monitor == MonitorEarlyStopping::ValidationLoss) {
            // Lower is better for loss
            if (currentScore < bestScore - minDelta) {
                bestScore = currentScore;
                wait = 0;
                if (isStore) {
                    this->Ws = Ws;
                    this->bs = bs;
                }
            } else {
                wait++;
            }

        } else {
            // Higher is better for accuracy
            if (currentScore > bestScore + minDelta) {
                bestScore = currentScore;
                wait = 0;
                if (isStore) {
                    this->Ws = Ws;
                    this->bs = bs;
                }
            } else {
                wait++;
            }
        }

        if (wait >= patience) {
            stoppedEpoch = epoch;
            return true;
        }

        return false;
    }

}   // namespace nn