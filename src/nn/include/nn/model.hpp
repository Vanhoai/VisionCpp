//
// Created by VanHoai on 9/6/25.
//

#ifndef MODEL_HPP
#define MODEL_HPP

// Libraries
#include <Eigen/Core>

// Sources
#include "layer.hpp"
#include "nn/early_stopping.hpp"
#include "nn/loss.hpp"
#include "nn/optimizer.hpp"

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
    virtual Eigen::MatrixXd feedforward(Eigen::MatrixXd &X) = 0;
    virtual void backpropagation(Eigen::MatrixXd &Y) = 0;
    virtual void update() = 0;
    virtual void fit(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int epochs, int batchSize,
                     bool verbose, int frequency, std::optional<EarlyStopping> earlyStopping) = 0;

    virtual Eigen::MatrixXd predict(Eigen::MatrixXd &X) = 0;

    virtual double calculateLoss(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) = 0;
    virtual double evaluate(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) = 0;

    virtual void load(const std::string &path) = 0;
    virtual void save(const std::string &path) = 0;
    virtual void summary() = 0;
};

class Sequential final : public Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss;
    std::unique_ptr<Optimizer> optimizer;

    int N, batchSize;
    Eigen::MatrixXd input, output;

public:
    Sequential(std::vector<std::unique_ptr<Layer>> &layers, std::unique_ptr<Loss> &loss,
               std::unique_ptr<Optimizer> &optimizer);

    Eigen::MatrixXd feedforward(Eigen::MatrixXd &X) override;
    void backpropagation(Eigen::MatrixXd &Y) override;
    void update() override;
    void fit(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int epochs, int batchSize, bool verbose,
             int frequency, std::optional<EarlyStopping> earlyStopping) override;

    Eigen::MatrixXd predict(Eigen::MatrixXd &X) override;

    double calculateLoss(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) override;
    double evaluate(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) override;

    void load(const std::string &path) override;
    void save(const std::string &path) override;
    void summary() override;

    friend std::ostream &operator<<(std::ostream &os, const Sequential &model);
};

}  // namespace nn

#endif  // MODEL_HPP
