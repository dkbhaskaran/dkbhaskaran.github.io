---
title : "Generic design in C++ - policy based design"
date: 2023-07-14 00:00:00 +0800
categories: [C++, OOP] # categories of each post are designed to contain up to two elements
tags: [ generic-design, cpp, policy-based-design, ooad, oop, object-oriented-design]  # TAG names should always be lowercase
math : true
description : Generic design in C++ - policy based design
toc: true
---

### Policy-Based Design

Policy-based design was popularized by **Andrei Alexandrescu** in his book *Modern C++ Design*. In this approach, a class's behavior is composed by combining multiple small, orthogonal classes called **policies**. Each policy encapsulates a specific aspect of behavior or configuration, allowing flexible and composable customization.

Instead of relying on multiple inheritance to mix in behaviors, policy-based design typically uses **templates** to achieve compile-time binding, providing **early type safety** and eliminating the ambiguity issues common in diamond inheritance.

This approach enables a combinatorial explosion of behavior variants without the complexity and overhead of runtime polymorphism.

#### Example Use Case: Customizing std::vector Allocation

Suppose we want to extend the memory allocation strategies of `std::vector` to support more advanced or customized behavior. We might define policies such as:
- **Growth** strategy : std::vector doesn't grow the storage as and when we try to push data in. It usually adopts a exponential growth strategy which can be sometime result in inefficient memory usage. Other strategy is linear growth 
- Different allocation pattern like **Pool-based** memory allocation or using std::malloc
- Adding instrumentation/debug to memory allocation policies. 

Here is an example of how we can use simple policy classes to create multitude of options for an allocator to std::vector.

```cpp

#include <iostream>
#include <list>
#include <vector>

// An allocator strategy using std::malloc and free
// To be compliant with std::vector, we may need to
// define the types as below and two other functions
// allocate and deallocate.

// Such allocator types can be extended to other types that provide
// thread safety or to use a custom memory allocation engine.
template <typename T> class MallocAllocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  T *allocate(std::size_t size) {
    return static_cast<T *>(std::malloc(sizeof(T) * size));
  }

  void deallocate(T *memory, size_t size = 0) {
    std::free(memory);
  }
};

// An allocator using pre-allocated pool memory.
// Usually the copy constructor and copy assignment operator
// implementation is not required. However std::vector::get_allocator
// needs to copy the allocator and hence it is required. In pool 
// allocator we cannot depend on the default one as it may result
// in double free of pool memory.

template <typename T> class PoolAllocator {
private:
  struct Chunk {
    T *start;
    size_t size;

    Chunk(T *s, std::size_t sz) : start(s), size(sz) {}
  };

public:
  T *pool;
  size_t pool_size;
  std::list<Chunk> free;
  std::list<Chunk> allocated;

  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  PoolAllocator(std::size_t size = 1024 * 1024)
      : pool_size(size) {
    pool = static_cast<T *>(std::malloc(pool_size));
    if (!pool) {
      throw std::bad_alloc();
    }
    free.push_back({pool, pool_size});
  }

  PoolAllocator(const PoolAllocator &other)
      : pool_size(other.pool_size) {
    pool = static_cast<T *>(std::malloc(pool_size));
  }

  PoolAllocator &operator=(const PoolAllocator &other) {
    pool_size = other.pool_size;
    pool = static_cast<T *>(std::malloc(pool_size));
  }

  ~PoolAllocator() {
    if (pool) {
      std::free(pool);
    }
    pool = nullptr;
  }
  T *allocate(size_t size) {
    int bytes = size * sizeof(T);
    for (auto it = free.begin(); it != free.end(); it++) {
      if (it->size >= bytes) {
        T *result = it->start;

        allocated.emplace_back(it->start, bytes);
        T *new_start = it->start + bytes;
        size_t new_size = it->size - bytes;

        free.erase(it);
        free.emplace_back(new_start, new_size);
        return result;
      }
    }

    return nullptr;
  }

  void deallocate(T *memory, size_t size = 0) {
    for (auto it = allocated.begin(); it != allocated.end();
         it++) {
      if (it->start == memory) {
        free.emplace_back(*it);
        allocated.erase(it);
        break;
        // Coalesce the adjacent chunks. Logic not implemented
      }
    }
  }
};

// Allocator adaptor for Instrumenting different allocator
// types. In here for demonstration purposes on two statistics
// are shown, which can be extended to collect across different
// std::vectors using static and atomic variables and so on.
template <typename T, typename AllocatorType>
class InstrumentedAllocator {
private:
  struct Stats {
    int allocations;
    int frees;

    Stats() { allocations = frees = 0; }
  };

  Stats stats;

public:
  AllocatorType allocator;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename... Args>
  InstrumentedAllocator(Args &&...args)
      : allocator(std::forward<Args>(args)...) {}

  T *allocate(size_t size) {
    stats.allocations++;
    return allocator.allocate(size);
  }

  void deallocate(T *memory, size_t size = 0) {
    stats.frees++;
    allocator.deallocate(memory, size);
  }

  void print_stat() {
    std::cout << "\nAllocs = " << stats.allocations << std::endl;
    std::cout << "Frees = " << stats.allocations << std::endl;
  }
};

int main() {
  using InstrumentedMallocAllocator =
      InstrumentedAllocator<int, MallocAllocator<int>>;
  using InstrumentedPoolAllocator =
      InstrumentedAllocator<int, PoolAllocator<int>>;

  std::vector<int, InstrumentedPoolAllocator> temp;
  for (auto i = 0; i < 5; i++) {
    temp.push_back(i);
  }

  // Contrary to one's expectation it need not print 5
  // as we are doing 5 push_backs. That is due to the reason
  // that std::vector grows exponentially and result in fewer
  // allocations and hence frees.
  temp.get_allocator().print_stat();
}

```

When decomposing a class into policies, aim to identify orthogonal policies — that is, policies that do not depend on or influence each other. If two policies interact, they are not orthogonal. Keeping policies orthogonal allows us to modify or replace them independently, improving flexibility and maintainability.
