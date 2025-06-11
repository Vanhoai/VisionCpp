//
// Created by VanHoai on 11/6/25.
//

#include "gtest/gtest.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int results = RUN_ALL_TESTS();
    return results;
}