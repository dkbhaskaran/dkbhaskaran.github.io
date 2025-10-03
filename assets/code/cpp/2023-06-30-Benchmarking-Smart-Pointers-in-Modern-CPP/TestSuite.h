#ifndef __TESTSUITE_H__
#define __TESTSUITE_H__

#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class SimpleTimer {
private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_time;

public:
  void start() { start_time = Clock::now(); }

  double getElapsedTime() {
    auto end_time = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time);
    return duration.count();
  }
};

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

// Note the use of anonymous namespace for eliminating the namespace
// clutter. As these structs will never be used directly, it is better
// to pack them in a anonymous namespace.
#define REGISTER_TESTCASE(TestCaseType)                                        \
  namespace {                                                                  \
  struct TestCaseType##Registrar {                                             \
    TestCaseType##Registrar() {                                                \
      TestSuiteRegistry::getInstance().addTestFactory<TestCaseType>();         \
    }                                                                          \
  };                                                                           \
  static TestCaseType##Registrar TestCaseType##Registrar;                      \
  }

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
      std::cout << result.first << "  :   " << result.second << " ns"
                << std::endl;
    }
  }
};

#endif // __TESTSUITE_H__
