//
// Created by VanHoai on 9/6/25.
//

#ifndef MODEL_HPP
#define MODEL_HPP

#include <Eigen/Core>

#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

using namespace std;
using namespace Eigen;

namespace nn {

    class Model {
        public:
            virtual ~Model() = default;

            /**
             * Function for train the model
             * 1. Forward pass: calculate the output of the model
             * 2. Backward pass: calculate the gradients
             * 3. Update the weights and biases
             */
            virtual MatrixXd feedforward(MatrixXd &X) = 0;
            virtual void backpropagation(MatrixXd &Y) = 0;
            virtual void update() = 0;
            virtual void fit(MatrixXd &X, MatrixXd &Y, int epochs,
                             int batchSize) = 0;

            virtual MatrixXd predict(MatrixXd &X) = 0;
            virtual double evaluate(MatrixXd &Y, MatrixXd &A) = 0;

            virtual void load(const string &path) = 0;
            virtual void save(const string &path) = 0;
    };

    class Sequential final : public Model {
        private:
            vector<unique_ptr<Layer>> layers;
            unique_ptr<Loss> loss;
            unique_ptr<Optimizer> optimizer;

            int N, batchSize;
            MatrixXd input, output;

        public:
            Sequential(vector<unique_ptr<Layer>> &layers,
                       unique_ptr<Loss> &loss,
                       unique_ptr<Optimizer> &optimizer);

            MatrixXd feedforward(MatrixXd &X) override;
            void backpropagation(MatrixXd &Y) override;
            void update() override;
            void fit(MatrixXd &X, MatrixXd &Y, int epochs,
                     int batchSize) override;

            MatrixXd predict(MatrixXd &X) override;
            double evaluate(MatrixXd &Y, MatrixXd &A) override;
            void load(const string &path) override;
            void save(const string &path) override;

            friend ostream &operator<<(ostream &os, const Sequential &model);
    };

}   // namespace nn

#endif   // MODEL_HPP
