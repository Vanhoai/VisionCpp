//
// File        : tensor.hpp
// Author      : Hinsun
// Date        : 2025-06-25
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef TENSOR_HPP
#define TENSOR_HPP

/**
 * @brief Header file for the Tensor class, which provides a generic implementation
 * of a tensor data structure.
 *
 * This class supports basic tensor operations such as creation, resizing, accessing
 * elements, and basic arithmetic operations.
 */

#include <vector>

namespace core {

    template <typename T>
    class RowProxy {
        public:
            RowProxy(T* row, const std::size_t cols) : row_(row), cols_(cols) {}

            T& operator[](std::size_t col) {
                if (col >= cols_)
                    throw std::out_of_range("Column index out of range");

                return row_[col];
            }

            const T& operator[](std::size_t col) const {
                if (col >= cols_)
                    throw std::out_of_range("Column index out of range");

                return row_[col];
            }

        private:
            T* row_;
            std::size_t cols_;
    };

    template <typename T>
    class Tensor {
        private:
            std::vector<size_t> shape_;
            std::unique_ptr<T[]> data_;
            size_t size_;

            static size_t calculateTotalSize(const std::vector<size_t>& shape) {
                size_t size = 1;
                for (const size_t dim : shape) size *= dim;
                return size;
            }

            [[nodiscard]] size_t calculateIndex(const std::vector<size_t>& indices) const {
                if (indices.size() != shape_.size())
                    throw std::invalid_argument("Number of indices must match tensor dimensions");

                size_t index = 0;
                size_t stride = 1;

                for (int i = shape_.size() - 1; i >= 0; --i) {
                    if (indices[i] >= shape_[i])
                        throw std::out_of_range("Index out of bounds");

                    index += indices[i] * stride;
                    stride *= shape_[i];
                }

                return index;
            }

        public:
            Tensor() : size_(0) {}

            /**
             * @brief Constructs a Tensor with the specified dimensions.
             *
             * The dimensions can be provided as a variadic template, allowing for
             * flexible tensor shapes.
             *
             * @tparam Dims The dimensions of the tensor.
             * @param dims The dimensions of the tensor.
             *
             * @example
             * core::Tensor<float> tensor(4, 5, 6); create a 3D tensor with shape (4, 5, 6)
             * core::Tensor<int> tensor(6, 7); create a 2D tensor with shape (6, 7)
             */
            template <typename... Dims>
            explicit Tensor(Dims... dims) {
                static_assert((std::is_convertible_v<Dims, size_t> && ...),
                              "All dimensions must be size_t-convertible");

                shape_ = std::vector<size_t>{static_cast<size_t>(dims)...};
                size_ = calculateTotalSize(shape_);
                data_ = std::make_unique<T[]>(size_);
                std::fill(data_.get(), data_.get() + size_, T{});
            }

            /**
             * @brief Constructs a Tensor with the specified shape.
             *
             * The shape is provided as a vector of size_t, allowing for dynamic tensor shapes.
             *
             * @param shape A vector representing the shape of the tensor.
             *
             * @example
             * core::Tensor<float> tensor({4, 5, 6}); create a 3D tensor with shape (4, 5, 6)
             * core::Tensor<int> tensor({6, 7}); create a 2D tensor with shape (6, 7)
             */
            explicit Tensor(const std::vector<size_t>& shape)
                : shape_(shape), size_(calculateTotalSize(shape)) {
                data_ = std::make_unique<T[]>(size_);
                std::fill(data_.get(), data_.get() + size_, T{});
            }

            /**
             * @brief Constructs a Tensor with the specified shape and initializes all elements to a
             * given value.
             *
             * The shape is provided as a vector of size_t, and all elements are initialized to the
             * specified value.
             *
             * @param shape A vector representing the shape of the tensor.
             * @param value The value to initialize all elements of the tensor.
             *
             * @example
             * core::Tensor<float> tensor({4, 5, 6}, 1.0f); create a 3D tensor with shape (4, 5, 6)
             * initialized to 1.0f core::Tensor<int> tensor({6, 7}, 0); create a 2D tensor with
             * shape (6, 7) initialized to 0
             */
            Tensor(const std::vector<size_t>& shape, const T& value)
                : shape_(shape), size_(calculateTotalSize(shape)) {
                data_ = std::make_unique<T[]>(size_);
                std::fill(data_.get(), data_.get() + size_, value);
            }

            explicit Tensor(const std::vector<std::vector<T>>& data) {
                const size_t rows = data.size();
                if (rows == 0) {
                    shape_ = {0, 0};
                    size_ = 0;
                    data_ = std::make_unique<T[]>(0);
                    return;
                }

                const size_t cols = data[0].size();
                shape_ = {rows, cols};
                size_ = rows * cols;
                data_ = std::make_unique<T[]>(size_);
                for (size_t i = 0; i < rows; ++i) {
                    if (data[i].size() != cols)
                        throw std::invalid_argument(
                            "All rows must have the same number of columns");

                    std::copy(data[i].begin(), data[i].end(), data_.get() + i * cols);
                }
            }

            static Tensor zeros(const std::vector<size_t>& shape) { return Tensor(shape, 0); }
            static Tensor ones(const std::vector<size_t>& shape) { return Tensor(shape, 1); }

            /**
             * @brief Copy constructor for the Tensor class.
             *
             * This constructor creates a new tensor as a copy of another tensor.
             *
             * @param other the tensor to copy from.
             */
            Tensor(const Tensor& other) : shape_(other.shape_), size_(other.size_) {
                data_ = std::make_unique<T[]>(size_);
                std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
            }

            /**
             * @brief Accessor for the tensor elements using a vector of indices.
             *
             * This operator allows you to access elements of the tensor using a vector of indices.
             *
             * @param indices a vector presenting the position of the element in the tensor.
             * @return T& a reference to the element at the specified indices.
             */
            T& operator()(const std::vector<size_t>& indices) {
                return data_[calculateIndex(indices)];
            }

            /**
             * @brief Const accessor for the tensor elements using a vector of indices.
             *
             * This operator allows you to access elements of the tensor using a vector of indices
             *
             * @param indices a vector presenting the position of the element in the tensor.
             * @return const T& a const reference to the element at the specified indices.
             */
            const T& operator()(const std::vector<size_t>& indices) const {
                return data_[calculateIndex(indices)];
            }

            template <typename... Indices>
            T& at(Indices... indices) {
                static_assert((std::is_convertible_v<Indices, size_t> && ...),
                              "All indices must be size_t-convertible");

                const std::vector<size_t> idx{static_cast<size_t>(indices)...};

                if (idx.size() != shape_.size())
                    throw std::invalid_argument("Number of indices must match tensor dimensions");

                return data_[calculateIndex(idx)];
            }

            template <typename... Indices>
            const T& at(Indices... indices) const {
                static_assert((std::is_convertible_v<Indices, size_t> && ...),
                              "All indices must be size_t-convertible");

                const std::vector<size_t> idx{static_cast<size_t>(indices)...};

                if (idx.size() != shape_.size())
                    throw std::invalid_argument("Number of indices must match tensor dimensions");

                return data_[calculateIndex(idx)];
            }

            /**
             * @brief Accessor for the tensor elements using row and column indices.
             *
             * This operator allows you to access elements of a 2D tensor using row and column
             * indices.
             *
             * @param row the row index of the element in the tensor.
             * @param col the column index of the element in the tensor.
             * @return T& a reference to the element at the specified row and column.
             */
            T& operator()(const std::size_t row, const std::size_t col) {
                if (shape_.size() != 2)
                    throw std::invalid_argument("Tensor is not 2-dimensional");

                return data_[row * shape_[1] + col];
            }

            /**
             * @brief Const accessor for the tensor elements using row and column indices.
             *
             * This operator allows you to access elements of a 2D tensor using row and column
             * indices.
             *
             * @param row the row index of the element in the tensor.
             * @param col the column index of the element in the tensor.
             * @return const T& a const reference to the element at the specified row and column.
             */
            const T& operator()(const std::size_t row, const std::size_t col) const {
                if (shape_.size() != 2)
                    throw std::invalid_argument("Tensor is not 2-dimensional");

                return data_[row * shape_[1] + col];
            }

            /**
             * @brief Assigns the values of another tensor to this tensor.
             *
             * This operator allows you to assign the values of another tensor to this tensor.
             *
             * @param other the tensor to assign from.
             * @return Tensor& a reference to this tensor after assignment.
             */
            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    shape_ = other.shape_;
                    size_ = other.size_;
                    data_ = std::make_unique<T[]>(size_);
                    std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
                }

                return *this;
            }

            /**
             * @brief Move assignment operator for the Tensor class.
             *
             * This operator allows you to move the contents of another tensor to this tensor,
             * transferring ownership of the data.
             *
             * @param other the tensor to move from.
             * @return Tensor& a reference to this tensor after the move assignment.
             */
            Tensor& operator=(Tensor&& other) noexcept {
                if (this != &other) {
                    shape_ = std::move(other.shape_);
                    data_ = std::move(other.data_);
                    size_ = other.size_;
                    other.size_ = 0;
                }
                return *this;
            }

            Tensor operator+(const Tensor& other) const {
                if (shape_ != other.shape_)
                    throw std::invalid_argument("Tensors must have the same shape for addition");

                Tensor result(shape_);
                for (size_t i = 0; i < size_; ++i) result.data_[i] = data_[i] + other.data_[i];

                return result;
            }

            Tensor operator-(const Tensor& other) const {
                if (shape_ != other.shape_)
                    throw std::invalid_argument("Shape mismatch in operator-");

                Tensor result(shape_);
                for (size_t i = 0; i < size_; ++i) result.data_[i] = data_[i] - other.data_[i];
                return result;
            }

            Tensor operator*(const Tensor& other) const {
                if (shape_ != other.shape_)
                    throw std::invalid_argument("Shape mismatch in operator*");

                Tensor result(shape_);
                for (size_t i = 0; i < size_; ++i) result.data_[i] = data_[i] * other.data_[i];
                return result;
            }

            Tensor operator/(const Tensor& other) const {
                if (shape_ != other.shape_)
                    throw std::invalid_argument("Shape mismatch in operator/");

                Tensor result(shape_);
                for (size_t i = 0; i < size_; ++i) {
                    if (other.data_[i] == T{})   // Check division by zero
                        throw std::domain_error("Division by zero in operator/");
                    result.data_[i] = data_[i] / other.data_[i];
                }
                return result;
            }

            static Tensor dot(const Tensor& A, const Tensor& B) {
                // if (A.shape() == 1 && A.shape())
            }

            /**
             * @brief Properties of the Tensor class.
             *
             * Below are all the properties of the Tensor class.
             * 1. dimensions() - Returns the number of dimensions of the tensor.
             * 2. shape() - Returns the shape of the tensor as a vector of size_t.
             */
            [[nodiscard]] size_t dimensions() const { return shape_.size(); }
            [[nodiscard]] const std::vector<size_t>& shape() const { return shape_; }
            [[nodiscard]] bool empty() const { return size_ == 0; }
            [[nodiscard]] size_t size() const { return size_; }
    };

}   // namespace core

#endif   // TENSOR_HPP
