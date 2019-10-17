#include <gtest/gtest.h>
#include "coordinates.h"


// To return the x value
TEST(TestGet, XValue) {
  Coordinates P(3.1,7.4);
  EXPECT_EQ(3.1, P.x_get());
}

// main
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
