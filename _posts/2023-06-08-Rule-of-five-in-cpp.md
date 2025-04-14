---
title : Rule of Five in C++
date: 2023-06-08
categories: [C++]
tags: [C++, OOP]
---

In C++, the **Rule of Five** states that if a class needs to define any of the following special member functions, it should explicitly define **all five**: the **destructor**, **copy constructor**, **copy assignment operator**, **move constructor**, and **move assignment operator**. This ensures proper management of resources and avoids issues like resource leaks or undefined behavior.

The **Rule of Three** existed before C++11, when **move semantics** weren’t defined, and only focused on the **destructor**, **copy constructor**, and **copy assignment operator**. With the introduction of move semantics in C++11, the rule expanded to five functions.

### Why define all five?

The compiler can automatically generate default versions of these functions. However, if one of them is defined, it generally means that the class manages a resource (like dynamic memory) that requires explicit handling. For example, if a class contains a pointer to dynamically allocated memory, the destructor must manually free that memory. This would also imply that:

- The **copy constructor** should allocate new memory for the new instance and copy the contents of the existing memory (deep copy).
- The **copy assignment operator** should also allocate new memory for the assignment and deep copy the contents from the source object.
- The **move constructor** and **move assignment operator** are used to efficiently transfer ownership of resources between objects.

Without defining all five functions, one of them might behave unexpectedly, leading to resource leaks or undefined behavior. So, the **Rule of Five** ensures consistency in how resources are managed.

---

## Demonstration of Rule of Five with `SharedResource`

Let's consider a class, **`SharedResource`**, that demonstrates **shared ownership** of a resource between instances of this Class. It uses reference counting for keeping track of all the instances of a class that uses this shared resource. For simplicity, we'll assume the reference count itself is the resource the class manages, though this design can easily be extended to manage more complex resources like a **file descriptor** or **socket**. Yes, this is very close to the `shared_ptr` concept in modern C++.

### Functional specification of `SharedResource` Class:

1. **Naming and Identification**:
   - Each instance of `SharedResource` should be identified by a unique **name**.

2. **Shared Ownership**:
   - Multiple instances can share ownership of the resource. Sharing of resource happens when a new instance is created using an existing instance, either by construction or assignment.

3. **Reference Count Management**:
   - Upon construction, the reference count should be set to **1** (indicating that the object is being used for the first time).
   - Any subsequent **copy constructor** or **assignment operator** should increment the reference count to reflect that another instance is sharing the resource.
   - When a `SharedResource` object is destroyed (goes out of scope), its destructor should decrement the reference count.
   - If the last reference to the object goes out of scope (i.e., reference count reaches 0), the reference count should be deleted (and the resource deallocated).
   - When a `SharedResource` object is moved, ownership of the reference count should be transferred to the new object, and the original object’s reference count should be decremented by one.

### Code implementation:

```cpp
#include <string>
#include <iostream>
#include <atomic>

class SharedResource {
public:
  // Constructor: Initialize with a name and set reference count to 1
  SharedResource(const std::string &resourceName)
      : name(resourceName), referenceCount(new std::atomic<int>(1)) {}

  // Destructor: Decrement reference count and delete when it reaches 0
  ~SharedResource() { releaseReferenceCount(); }

  // Copy Constructor: Increment reference count
  SharedResource(const SharedResource &other)
      : name(other.name), referenceCount(other.referenceCount) {
    ++(*referenceCount);
    std::cout << name << " : Copy constructor, refCount: " << *referenceCount
              << std::endl;
  }

  // Copy Assignment Operator: Decrement old reference count, copy new reference
  SharedResource &operator=(const SharedResource &other) {
    if (this == &other)
      return *this; // Handle self-assignment

    releaseReferenceCount();

    referenceCount = other.referenceCount;
    name = other.name;
    ++(*referenceCount);
    std::cout << name << " : Copy assignment, refCount: " << *referenceCount
              << std::endl;
    return *this;
  }

  // Move Constructor: Transfer reference count ownership
  SharedResource(SharedResource &&other) noexcept
      : name(std::move(other.name)), referenceCount(other.referenceCount) {
    other.referenceCount = nullptr;
    std::cout << name << " : Move constructor, refCount: " << *referenceCount
              << std::endl;
  }

  // Move Assignment Operator: Transfer reference count ownership
  SharedResource &operator=(SharedResource &&other) noexcept {
    if (this == &other)
      return *this; // Handle self-assignment

    releaseReferenceCount();

    referenceCount = other.referenceCount;
    name = std::move(other.name);
    other.referenceCount = nullptr;
    std::cout << name << " : Move assignment, refCount: " << *referenceCount
              << std::endl;
    return *this;
  }

private:
  // Helper function to decrement reference count
  void releaseReferenceCount() {
    if (referenceCount && referenceCount->fetch_sub(1) == 1) {
      delete referenceCount;
      referenceCount = nullptr;
      std::cout << "Reference count deleted" << std::endl;
    }
  }

  std::string name;                 // The name of the resource
  std::atomic<int> *referenceCount; // Reference count (shared among instances)
};
```

### Some Notable Details:

- The class has two attributes: `name` and `referenceCount` of types `std::string` and `std::atomic<int>`.
- **Constructor**: Initializes the `referenceCount` to 1, indicating that one instance of the object is using this shared resource.
- **Copy Constructor**: Simply increments the reference count, showing that one more instance is added which uses this shared resource.
- **Copy Assignment**: Decrements the reference count in the assignee and deletes the `referenceCount` if the value has reached 0. Then it ensures that the `referenceCount` pointer is now used by this class and increments that `referenceCount` by 1 to indicate that there is one more instance using the shared resource.
- **Move Constructor**: Transfers the `referenceCount` ownership. At the end, the `referenceCount` in the source is nullified to ensure safety. The `referenceCount` is not incremented since only ownership transfer happens.
- **Move Assignment**: Similar to copy assignment, it reduces and deletes the assignee’s `referenceCount` if required, and uses the source’s `referenceCount` in the assignee. There is no increment of `referenceCount` since it is just transferring ownership.
- **Const Arguments**: The source is marked as `const` in the copy constructor and copy assignment because we do not modify the source, and it is generally a good practice to mark such arguments as `const`. This is not the case in move constructor and assignment operators.
- **noexcept**: The move constructor and move assignment are marked `noexcept` primarily for the following reasons:
    - It allows the compiler to perform more aggressive optimization.
    - It enables STL containers to use move semantics in resizing operations when move is marked `noexcept`, which otherwise would force `std::vector` to use copy semantics for strong exception safety.

### Conclusion:

The **Rule of Five** ensures that your class properly manages resources, particularly when dealing with ownership and reference counting. By explicitly defining the five special member functions (destructor, copy constructor, copy assignment operator, move constructor, and move assignment operator), the class can effectively share, copy, and move resources in a safe and efficient manner. 
