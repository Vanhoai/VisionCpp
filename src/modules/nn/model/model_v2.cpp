//
// Created by VanHoai on 10/6/25.
//

#include "model.hpp"
#include <iomanip>

namespace nn {

    void drawHorizontalLine(std::ostream &os, const std::vector<int> &widths,
                            const std::string &corner = "+",
                            const std::string &horizontal = "-",
                            const std::string &junction = "+") {
        os << corner;
        for (size_t i = 0; i < widths.size(); ++i) {
            for (int j = 0; j < widths[i] + 2; ++j) {   // +2 for padding
                os << horizontal;
            }
            if (i < widths.size() - 1) {
                os << junction;
            }
        }
        os << corner << "\n";
    }

    // Helper function to draw a table row
    void drawTableRow(std::ostream &os, const std::vector<std::string> &data,
                      const std::vector<int> &widths,
                      const std::string &separator = "|") {
        os << separator;
        for (size_t i = 0; i < data.size() && i < widths.size(); ++i) {
            os << " " << std::left << std::setw(widths[i]) << data[i] << " "
               << separator;
        }
        os << "\n";
    }

    std::ostream &operator<<(std::ostream &os, const Sequential &model) {
        // Determine maximum widths for each column
        int nameWith = 12;
        int inputWidth = 8;
        int outputWidth = 8;
        int paramWith = 16;

        for (const auto &layer : model.layers) {
            std::string layerName = layer->getName();
            nameWith = std::max(nameWith, static_cast<int>(layerName.length()));

            inputWidth = std::max(
                inputWidth,
                static_cast<int>(
                    std::to_string(layer->getInputDimension()).length()));

            outputWidth = std::max(
                outputWidth,
                static_cast<int>(
                    std::to_string(layer->getOutputDimension()).length()));
        }

        // Adjust widths with padding
        nameWith = std::min(nameWith + 2, 20);
        inputWidth += 2;
        outputWidth += 2;
        paramWith += 2;

        const string title = "SEQUENTIAL MODEL";
        const int totalWidth = nameWith + inputWidth + outputWidth + paramWith +
                               10;   // 10 = 4 "No" and 6 "+|"
        const int preSpace = totalWidth / 2 - title.length() / 2;
        const int posSpace = totalWidth - preSpace - title.length() - 2;

        os << "\n";
        os << "+" << std::string(totalWidth - 2, '-') << "+\n";
        os << "|" << std::string(preSpace, ' ') << title
           << std::string(posSpace, ' ') << "|\n";
        os << "+" << std::string(totalWidth - 2, '-') << "+\n";

        // Table structure
        const std::string separator = "+----+" + std::string(nameWith, '-') +
                                      "+" + std::string(inputWidth, '-') + "+" +
                                      std::string(outputWidth, '-') + "+" +
                                      std::string(paramWith, '-') + "+";

        os << "| No |" << std::setw(nameWith) << std::left << " Layer"
           << "|" << std::setw(inputWidth) << " Input" << "|"
           << std::setw(outputWidth) << " Output" << "|" << std::setw(paramWith)
           << " Parameters" << "|\n";

        os << separator << "\n";

        for (size_t i = 0; i < model.layers.size(); ++i) {
            const auto &layer = *model.layers[i];
            std::string layerType = layer.getName();

            const int params =
                layer.getInputDimension() * layer.getOutputDimension() +
                layer.getOutputDimension();   // W + b

            os << "| " << std::left << "0" << (i + 1) << " |";
            os << " " << std::setw(nameWith - 1) << std::left << layerType
               << "|"
               << " " << std::setw(inputWidth - 1) << std::left
               << layer.getInputDimension() << "|"
               << " " << std::setw(outputWidth - 1) << std::left
               << layer.getOutputDimension() << "|"
               << " " << std::setw(paramWith - 1) << std::left << params
               << "|\n";

            if (i < model.layers.size() - 1)
                os << separator << "\n";
        }

        os << separator << "\n";
        return os;
    }

}   // namespace nn