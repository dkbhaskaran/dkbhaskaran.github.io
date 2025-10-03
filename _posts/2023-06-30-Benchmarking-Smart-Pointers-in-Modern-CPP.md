---
title : Benchmarking Smart Pointers in Modern C++
date: 2023-06-30
categories: [C++]
tags: [C++, OOP, Smart Pointers]
---

# Smart Pointers in C++

Smart pointers in C++ help users manage dynamically allocated objects safely and efficiently. While raw pointers are inherently supported in C++, they come with several pitfalls such as memory leaks, double deletes, and dangling pointers. The smart pointer implementations in the C++ standard library act as wrappers around raw pointers, eliminating many common issues associated with manual memory management.

There are four main types of smart pointers:

- **`std::auto_ptr`**:  
  This was a simple wrapper around a raw pointer that automatically deleted the managed object when it went out of scope. However, it was deprecated and removed in C++17 due to its problematic copy semantics.

- **`std::unique_ptr`**:  
  This smart pointer provides exclusive ownership of a resource. Ownership can be transferred (moved) but not copied. Like `auto_ptr`, it automatically deletes the object when the `unique_ptr` goes out of scope.

- **`std::shared_ptr`**:  
  Unlike `unique_ptr`, a `shared_ptr` allows multiple owners of the same resource. It maintains a reference count of how many `shared_ptr` instances point to the object. When the count drops to zero, the object is automatically destroyed.

- **`std::weak_ptr`**:  
  A `weak_ptr` provides a non-owning, non-reference-counting access to an object managed by one or more `shared_ptr`s. The existence or destruction of a `weak_ptr` does not affect the lifetime of the managed object.

  Use cases for `weak_ptr` include:  
  - Acting as a non-owning observer to a resource without extending its lifetime.  
    This differs from using a raw pointer directly because the managed object may be deleted while the raw pointer still exists, leading to undefined behavior. With `weak_ptr`, you can safely check the object’s validity using `weak_ptr.lock()`.  
  - Helping to break circular dependencies between `shared_ptr`s. If two `shared_ptr`s hold references to each other, they can create a cyclic dependency preventing proper destruction. `weak_ptr` can be used to avoid this scenario. Consider the below program

```cpp
#include <iostream>
#include <memory>

class B;

class A {
public:
  std::shared_ptr<B> b;

  ~A() {
    std::cout << "A destroyed" << std::endl;
  }
};

class B {
public:
  std::shared_ptr<A> a;
  ~B() {
    std::cout << "B destroyed" << std::endl;
  }
};

int main() {
  std::shared_ptr<A> a = std::make_shared<A>();
  std::shared_ptr<B> b = std::make_shared<B>();

  a->b = b;
  b->a = a;

  std::cout << "porgram ending" << std::endl;

  /* When this frame ends, ideally destructors for A and B,
   * should be called, but they are not called as each of
   * these maintains a shared_ptr of other class.
   *
   * Instead we should use weak_ptr
   * class B {
   * public:
   *   std::weak_ptr<A> a;
   * };
   *
   * and in main
   * b->a = a;
   */
  return 0;
}
```

Well, it is quite obvious when to use each type of smart pointer. But sometimes, in terms of performance, which one is better? The answer is, of course, `std::unique_ptr`, which is faster than `std::shared_ptr` because the latter needs to maintain an atomic reference count. But how much faster? 

The later part of this article explores this question by following elaborate design considerations as if for a real application. At a high level, we will compare smart pointer performance. This system should help developers choose the right pointer type, and the results should be easy to understand.

> **Note:** We could use existing testing tools like gtest or cppUnit, but to explore the design aspects, let's design our own benchmark.

---

### Requirement Analysis

Here are some questions and answers to clarify and elaborate the high-level design:

- **Scope and Contextual Questions**
  - Why do we need this project?  
    For educational purposes.
  - Are we optimizing for a specific domain like gaming or HPC?  
    No, just a generic domain.
  - Who are the target audiences of the results? Testing team or developers?  
    Developers or anyone interested in learning.
  - Do we include `weak_ptr`?  
    No, only ownership pointers are included for now.
  - What compiler optimization levels should be used?  
    No optimization.
  - What object sizes should this test?  
    1 integer and 1000 integers.

- **Performance Requirements**
  - How fast should the benchmark execute?  
    Less than 5 minutes.
  - How accurate should the measurement be?  
    Use `std::chrono::steady_clock::now`.

- **Other Considerations**
  - Should this program be thread-safe?  
    No.
  - Should this be portable across platforms?  
    No.
  - Are custom deleters part of this assignment?  
    No.

---

### System-Level Questions to Consider

These questions reflect how professionals think before starting a project:

- What is the actual goal of this project, and how will developers use it?
- Where will this be deployed (e.g., in a CI/CD platform)?
- Are there existing performance tools that this will complement?
- What is the actionable outcome from this exercise? How will it integrate into the overall workflow?
- What is the validation strategy? How do we verify that our tool measures what we need?  
  Statistical methods may be required for reliable measurement.

---

### Assumptions

At this stage, we list all assumptions for the project. Ideally, these should be agreed upon by the project stakeholders.

- **Compilers**  
  - We will use C++17.  
  - Avoid compiler optimizations by using `volatile` or memory barriers where needed.

- **Testing and Validation**  
  - Configurable iteration parameters with warm-up runs to determine how many times the test program executes.

---

### Test Scenarios

- Creation and deletion of `unique_ptr`, `shared_ptr`, and raw pointers.
- Copy operations: move for `unique_ptr`, copy for `shared_ptr` and raw pointers.
- Access patterns: dereferencing, array access, etc.

## Class Specification

These are the involved objects:

#### class TestCase

The `TestCase` class defines the abstract interface and lifecycle of a single test. It is intended to be subclassed by users to implement specific performance or functional tests. Each test case is configurable using parameters from `TestConfig`.

```cpp
struct TestConfig {
  int warmupRuns = 1000;
  int numIterations = 100000;
};

class TestCase {
  std::string name;
  TestConfig config;

public:
  TestCase(const TestConfig &cfg, const std::string &testName)
      : config(cfg), name(testName) {}
  virtual ~TestCase() = default;
  virtual std::string getName() const { return name; }
  // Do nothing in here.
  virtual void run() {}
  virtual void setUp() {}
  virtual void tearDown() {}
  virtual double getResults() const { return 0.0; }
};
```

#### Class TestSuiteRegistry

The `TestSuiteRegistry` acts as the central registry and factory manager for all test cases in the system. It is responsible for:  
- Collecting all registered test case types.  
- Creating instances of these tests on demand using the provided configuration.  
- Providing a single access point (`getInstance()`) via the Singleton pattern.

The `REGISTER_TESTCASE` macro offers a convenient and automatic way to register new test case types with the `TestSuiteRegistry`, eliminating the need for manual registration logic in `main()` or other parts of the codebase.

```cpp
class TestSuiteRegistry {
private:
  TestConfig config;
  using FactoryFunc =
      std::function<std::unique_ptr<TestCase>(const TestConfig &)>;

  std::vector<FactoryFunc> testFactory;

public:
  template <typename TestCaseType> void addTestFactory() {
    testFactory.push_back([](const TestConfig &config) {
      return std::make_unique<TestCaseType>(config);
    });
  }

  static TestSuiteRegistry &getInstance() {
    static TestSuiteRegistry suite;
    return suite;
  }

  std::vector<std::unique_ptr<TestCase>>
  initTestCases(const TestConfig &config) {
    std::vector<std::unique_ptr<TestCase>> testCases;
    for (auto factory : testFactory) {
      testCases.emplace_back(factory(config));
    }
    return testCases;
  }
};

#define REGISTER_TESTCASE(TestCaseType)                                        \
  namespace {                                                                  \
  struct TestCaseType##Registrar {                                             \
    TestCaseType##Registrar() {                                                \
      TestSuiteRegistry::getInstance().addTestFactory<TestCaseType>();         \
    }                                                                          \
  };                                                                           \
  static TestCaseType##Registrar TestCaseType##Registrar;                      \
  }
```

#### Class TestSuite

The `TestSuite` class serves as the **runtime driver** responsible for executing all registered test cases. It instantiates all test cases from the registry and runs them.

```cpp
class TestSuite {
private:
  TestConfig config;
  std::vector<std::unique_ptr<TestCase>> testCases;
  std::map<std::string, double> results;

  void initializeTestCases() {
    testCases = TestSuiteRegistry::getInstance().initTestCases(config);
  }

public:
  TestSuite(const TestConfig &cfg) : config(cfg) { initializeTestCases(); }

  void run() {
    for (auto &test : testCases) {
      std::cout << "Running Test : " << test->getName() << ": ";
      try {
        test->setUp();
        test->run();
        test->tearDown();
        std::cout << " Done" << std::endl;
        results[test->getName()] = test->getResults();
      } catch (const std::exception &e) {
        std::cout << " Failed with " << e.what() << std::endl;
      }
    }

    std::cout << "Results..." << "\n";
    for (auto result : results) {
      std::cout << result.first << "  :   " << result.second << " us"
                << std::endl;
    }
  }
};
```

### Actual TestCases

Now that we have the infrastructure to build test cases, let's move on to implementing the actual tests. Here, we show an example of an access test for `std::unique_ptr` and `std::shared_ptr`. 

The class that performs this test is a generic class capable of testing access patterns for different types of objects as well as different object sizes. We use two different types of objects here, **Light** and **Heavy**, based on their sizes. These are defined as follows:

```cpp 
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
```

The generic classes for the Light and Heavy test cases are defined as follows:

```cpp
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
```

**Note:** With this abstraction, it is easy to add different test case scenarios. Finally, we need to register these test cases with the test suite and execute them. Registration should be done in a `.cpp` file as shown below:

```cpp
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
```

On my system, this produces output like the following. We see that our earlier assumption about unique_ptr being faster is confirmed. Specifically, access is about 5% faster for heavy objects, while for small objects it is 44% faster — quite impressive! This benchmark was run with no compiler optimizations enabled. The approach can be easily extended to other scenarios like creation/deletion or copy/move operations, as discussed earlier.

```bash 
➜  smart_pointer_benchmark git:(main) ✗ g++ -std=c++17 main.cpp
➜  smart_pointer_benchmark git:(main) ✗ ./a.out                
Running Test : LightUniquePtrAllocationTest:  Done
Running Test : LightSharedPtrAllocationTest:  Done
Running Test : HeavySharedPtrAllocationTest:  Done
Running Test : HeavyUniquePtrAllocationTest:  Done
Results...
HeavySharedPtrAllocationTest  :   7.93954e+07 ns
HeavyUniquePtrAllocationTest  :   7.466e+07 ns
LightSharedPtrAllocationTest  :   7.87385e+06 ns
LightUniquePtrAllocationTest  :   4.40204e+06 ns
```

Hope this was a fruitful exercise in exploring the design process of a generic application and understanding the performance characteristics of smart pointers. The complete code is available at [this link](https://github.com/dkbhaskaran/dkbhaskaran.github.io/tree/main/assets/code/cpp/2023-06-30-Benchmarking-Smart-Pointers-in-Modern-CPP).

