#ifndef __BENCHMARKS_H__
#define __BENCHMARKS_H__

#include "TestSuite.h"
#include <algorithm>
#include <memory>

struct LightObject {
  int data;
  LightObject() : data(42) {}
  int *getData() { return &data; }
};

struct HeavyObject {
  std::vector<int> data;

  HeavyObject() : data(1000, 42) {}
  int *getData() { return &data[0]; }
};

template <typename SmartPtrType, typename ObjectType>
class AllocationBenchmark : public TestCase {
private:
  TestConfig config;
  std::string name;
  double testTime = 0.0;

public:
  explicit AllocationBenchmark(TestConfig _config, std::string _name)
      : TestCase(_config, _name) {}

  void setUp() override {
    if (config.warmupRuns <= 0) {
      return;
    }

    for (auto i = 0; i < config.warmupRuns; i++) {
      auto ptr = createPtrType();

      // Touch ptr to avoid optimization;
      *ptr->getData() = 42;
    }
  }

  void run() override {
    SimpleTimer timer;

    timer.start();
    for (auto i = 0; i < config.numIterations; i++) {
      auto ptr = createPtrType();
      *ptr->getData() = 42;
    }
    testTime = timer.getElapsedTime();
  }

  double getResults() const override { return testTime; }

private:
  SmartPtrType createPtrType() {
    if constexpr (std::is_same_v<SmartPtrType, std::unique_ptr<ObjectType>>) {
      return std::make_unique<ObjectType>();
    } else if constexpr (std::is_same_v<SmartPtrType,
                                        std::shared_ptr<ObjectType>>) {
      return std::make_shared<ObjectType>();
    }
  }
};

class LightUniquePtrAllocationTest
    : public AllocationBenchmark<std::unique_ptr<LightObject>, LightObject> {
public:
  LightUniquePtrAllocationTest(const TestConfig &config)
      : AllocationBenchmark(config, "LightUniquePtrAllocationTest") {}
};

class LightSharedPtrAllocationTest
    : public AllocationBenchmark<std::shared_ptr<LightObject>, LightObject> {
public:
  LightSharedPtrAllocationTest(const TestConfig &config)
      : AllocationBenchmark(config, "LightSharedPtrAllocationTest") {}
};

class HeavySharedPtrAllocationTest
    : public AllocationBenchmark<std::shared_ptr<HeavyObject>, HeavyObject> {
public:
  HeavySharedPtrAllocationTest(const TestConfig &config)
      : AllocationBenchmark(config, "HeavySharedPtrAllocationTest") {}
};
class HeavyUniquePtrAllocationTest
    : public AllocationBenchmark<std::unique_ptr<HeavyObject>, HeavyObject> {
public:
  HeavyUniquePtrAllocationTest(const TestConfig &config)
      : AllocationBenchmark(config, "HeavyUniquePtrAllocationTest") {}
};

#endif // __BENCHMARKS_H__
