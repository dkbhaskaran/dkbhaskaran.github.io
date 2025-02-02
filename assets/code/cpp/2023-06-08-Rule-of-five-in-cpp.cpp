#include <atomic>
#include <iostream>

class SharedResource {
public:
  // Constructor: Initializes reference count to 1
  SharedResource(std::string const &resourceName)
      : name{resourceName}, referenceCount{new std::atomic<int>(1)} {}

  // Destructor: Cleans up the reference count if it's no longer in use
  ~SharedResource() { releaseReferenceCount(); }

  // Copy constructor: Increments the reference count
  SharedResource(SharedResource const &other)
      : name(other.name), referenceCount(other.referenceCount) {
    ++(*referenceCount);
    std::cout << name << " : Shared resource copy constructor with refCount : "
              << *referenceCount << std::endl;
  }

  // Copy assignment: Decrements the current reference count, then increments
  // the new one
  SharedResource &operator=(const SharedResource &other) {

    // Handle self assignment
    if (this == &other) {
      return *this;
    }

    // Clean up the current object
    releaseReferenceCount();

    // Perform the assignment
    referenceCount = other.referenceCount;
    name = other.name;

    // Increment the reference count
    ++(*referenceCount);

    std::cout << name << " : Shared resource copy assignment with refCount : "
              << *referenceCount << std::endl;
    return *this;
  }

  // Move constructor : Transfers ownership of referenceCount
  SharedResource(SharedResource &&other) noexcept
      : name{std::move(other.name)}, referenceCount{other.referenceCount} {

    // Moving referenceCount is not necessary as this is just a pointer.
    other.referenceCount = nullptr;
    std::cout << name << " : Shared resource move constructor with refCount : "
              << *referenceCount << std::endl;
  }

  // Move assignment: Transfers ownership of referenceCount
  SharedResource &operator=(SharedResource &&other) noexcept {
    // Avoid self assignment
    if (this == &other) {
      return *this;
    }

    // Update the current refCount. Ensure the current objects, ref count is
    // decremented before overwriting it.
    releaseReferenceCount();

    referenceCount = other.referenceCount;
    name = std::move(other.name);
    other.referenceCount = nullptr;
    std::cout << name << " : Shared resource move assignment with refCount :"
              << *referenceCount << std::endl;

    return *this;
  }

private:
  void releaseReferenceCount() {
    // Helper function to decrement the reference count and delete if necessary
    if (referenceCount && referenceCount->fetch_sub(1) == 1) {
      delete referenceCount;
      referenceCount = nullptr;
      std::cout << "Reference count deleted" << std::endl;
    }
  }

  std::string name;
  std::atomic<int> *referenceCount;
};

int main() {

  SharedResource resource1("res1"); // Constructor

  SharedResource resource2{resource1}; // Copy constructor

  SharedResource resource3 = resource1; // Copy assignment

  SharedResource resrouce4{std::move(resource1)}; // Move constructor.

  SharedResource resrouce5 = std::move(resource2); // Move assignment.

  return 0;
}
