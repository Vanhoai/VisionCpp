//
// Created by VanHoai on 11/6/25.
//

#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <Eigen/Core>

using namespace Eigen;

namespace utilities {
    class Datasets {
        private:
            MatrixXd X;
            MatrixXd Y;
            int N;
            int classes;
            bool isOneHot;

        public:
            virtual ~Datasets() = default;
            Datasets(const int N, const int classes,
                     const bool isOneHot = false)
                : N(N), classes(classes), isOneHot(isOneHot) {
                this->X = MatrixXd::Zero(N, classes);
                this->Y = MatrixXd::Zero(N, classes);
            }

            virtual bool load() = 0;
            virtual void splits(MatrixXd &XTrain, MatrixXd &YTrain,
                                MatrixXd &XTest, MatrixXd &YTest) = 0;
    };

    class TwoDimensionDataset final : public Datasets {
        public:
            TwoDimensionDataset(const int N, const int classes,
                                const bool isOneHot = false)
                : Datasets(N, classes, isOneHot) {}

            bool load() override;
            void splits(MatrixXd &XTrain, MatrixXd &YTrain, MatrixXd &XTest,
                        MatrixXd &YTest) override;
    };

    class MNistDataset final : public Datasets {
        public:
            MNistDataset(const int N, const int classes,
                         const bool isOneHot = false)
                : Datasets(N, classes, isOneHot) {}

            bool load() override;
            void splits(MatrixXd &XTrain, MatrixXd &YTrain, MatrixXd &XTest,
                        MatrixXd &YTest) override;
    };

}   // namespace utilities

#endif   // DATASETS_HPP
