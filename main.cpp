#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Neural Networks
#include "src/modules/nn/loss/loss.hpp"
#include "src/modules/nn/model/model.hpp"
#include "src/modules/nn/optimizer/optimizer.hpp"
#include "src/modules/nn/prepare/prepare.hpp"
#include "src/nn/include/layer.hpp"

// Core
#include "src/core/utilities/multivariate_normal/multivariate_normal.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace nn;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>>
readIDX3UByteFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);
    std::cout << static_cast<int>(numImagesBytes[0]) << "  "
              << static_cast<int>(numImagesBytes[1]) << "  "
              << (int) static_cast<unsigned char>(numImagesBytes[2]) << "  "
              << static_cast<int>(numImagesBytes[3]) << "  " << std::endl;

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) |
                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) |
                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) |
                    static_cast<unsigned char>(numImagesBytes[3]);
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) |
                  (static_cast<unsigned char>(numRowsBytes[1]) << 16) |
                  (static_cast<unsigned char>(numRowsBytes[2]) << 8) |
                  static_cast<unsigned char>(numRowsBytes[3]);
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) |
                  (static_cast<unsigned char>(numColsBytes[1]) << 16) |
                  (static_cast<unsigned char>(numColsBytes[2]) << 8) |
                  static_cast<unsigned char>(numColsBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(numRows * numCols);
        file.read((char *) (image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>>
readLabelFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) |
                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) |
                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) |
                    static_cast<unsigned char>(numImagesBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(1);
        file.read((char *) (image.data()), 1);

        images.push_back(image);
    }

    file.close();

    return images;
}

void read_mnist() {
    string X_train_path = "/Users/aurorastudyvn/Workspace/ML/VisionCpp/"
                          "train-images-idx3-ubyte:train-images.idx3-ubyte";
    string y_train_path = "/Users/aurorastudyvn/Workspace/ML/VisionCpp/"
                          "train-labels-idx1-ubyte:train-labels.idx1-ubyte";

    std::vector<std::vector<unsigned char>> imagesFile =
        readIDX3UByteFile(X_train_path);
    std::vector<std::vector<unsigned char>> labelsFile =
        readLabelFile(y_train_path);
    std::vector<Mat> imagesData;   // Store your images
    std::vector<int> labelsData;   // Corresponding labels

    for (int imgCnt = 0; imgCnt < (int) imagesFile.size(); imgCnt++) {
        int rowCounter = 0;
        int colCounter = 0;

        Mat tempImg = Mat::zeros(Size(28, 28), CV_8UC1);
        for (int i = 0; i < (int) imagesFile[imgCnt].size(); i++) {

            tempImg.at<uchar>(Point(colCounter++, rowCounter)) =
                (int) imagesFile[imgCnt][i];
            if ((i) % 28 == 0) {
                rowCounter++;
                colCounter = 0;
                if (i == 756)
                    break;
            }
        }
        std::cout << (int) labelsFile[imgCnt][0] << std::endl;

        imagesData.push_back(tempImg);
        labelsData.push_back((int) labelsFile[imgCnt][0]);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    constexpr int N = 10000;
    int d = 2;
    int d1 = 5;
    int classes = 4;

    MatrixXd X(N, d);
    MatrixXd Y(N, classes);
    prepare(N, classes, d, X, Y);

    constexpr int T = N * 0.8;
    MatrixXd XTrain = X.block(0, 0, T, d);
    MatrixXd YTrain = Y.block(0, 0, T, classes);
    MatrixXd XTest = X.block(T, 0, N - T, d);
    MatrixXd YTest = Y.block(T, 0, N - T, classes);

    vector<unique_ptr<Layer>> layers;
    layers.push_back(make_unique<ReLU>(d, d1));
    layers.push_back(make_unique<Softmax>(d1, classes));

    unique_ptr<Loss> loss = make_unique<CrossEntropyLoss>();
    unique_ptr<Optimizer> optimizer = make_unique<SGD>(1e-1, 0.9, true, 0.0);
    nn::Sequential sequential(layers, loss, optimizer);

    constexpr int epochs = 30;
    constexpr int batchSize = 256;
    sequential.fit(XTrain, YTrain, epochs, batchSize);

    MatrixXd O = sequential.predict(XTest);
    const double accuracy = sequential.evaluate(YTest, O);
    cout << "Test Accuracy: " << accuracy * 100.0 << "%" << endl;

    return EXIT_SUCCESS;
}
