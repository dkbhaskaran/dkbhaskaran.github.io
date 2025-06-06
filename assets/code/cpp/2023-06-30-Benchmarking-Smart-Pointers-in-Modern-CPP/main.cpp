#include "Benchmarks.h"
#include "TestSuite.h"

REGISTER_TESTCASE(LightUniquePtrAllocationTest);
REGISTER_TESTCASE(LightSharedPtrAllocationTest);
REGISTER_TESTCASE(HeavySharedPtrAllocationTest);
REGISTER_TESTCASE(HeavyUniquePtrAllocationTest);

int main() {
  TestSuite ts(TestConfig{});
  ts.run();
  return 0;
}
