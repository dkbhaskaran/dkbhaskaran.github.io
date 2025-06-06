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
