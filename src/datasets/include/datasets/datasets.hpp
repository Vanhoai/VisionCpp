//
// Created by VanHoai on 11/6/25.
//

#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <Eigen/Core>

namespace datasets {
    /**
     * Dataset is class that provides a common interface for datasets.
     * Matrix X: features of the dataset, shape (N, d)
     * Vector y: labels of the dataset, shape (N,)
     */
    class Dataset {
        private:
            Eigen::MatrixXd X;
            Eigen::VectorXd Y;

            int N;
            int classes;
            bool isCached;
            bool isSetup;

        public:
            virtual ~Dataset() = default;
            Dataset(const int N, const int classes, const bool isCached = true)
                : N(N), classes(classes), isCached(isCached) {
                this->X = Eigen::MatrixXd::Zero(N, classes);
                this->Y = Eigen::VectorXd::Zero(N);
                this->isSetup = false;
            }

            [[nodiscard]] int getClasses() const { return classes; }
            [[nodiscard]] int getN() const { return N; }
            [[nodiscard]] bool getIsCached() const { return isCached; }

            [[nodiscard]] bool getIsSetup() const { return isSetup; }
            void setIsSetup(const bool isSetup) { this->isSetup = isSetup; }

            [[nodiscard]] Eigen::MatrixXd &getX() { return X; }
            void setX(const Eigen::MatrixXd &X) { this->X = X; }

            [[nodiscard]] Eigen::VectorXd &getY() { return Y; }
            void setY(const Eigen::VectorXd &Y) { this->Y = Y; }

            bool load(Eigen::MatrixXd &XTrain, Eigen::MatrixXd &YTrain, Eigen::MatrixXd &XTest,
                      Eigen::MatrixXd &YTest, int percentTrain, bool isShuffle) const;

            bool load(Eigen::MatrixXd &XTrain, Eigen::VectorXd &YTrain, Eigen::MatrixXd &XTest,
                      Eigen::VectorXd &YTest, int percentTrain, bool isShuffle) const;
    };

    class TwoDimensionDataset final : public Dataset {
        public:
            TwoDimensionDataset(const int N, const int classes, const bool isCached = false)
                : Dataset(N, classes, isCached) {}

            void setup(const Eigen::MatrixXd &mean, const Eigen::MatrixXd &covariance);
    };

    class MNistDataset final : public Dataset {
        public:
            MNistDataset(const int N, const int classes) : Dataset(N, classes) {}
            void setup();
    };

}   // namespace datasets

#endif   // DATASETS_HPP
