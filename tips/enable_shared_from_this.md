# enable_shared_from_this

Bad case:

```cpp
#include <iostream>
#include <memory>

class KK {
    public:
        std::shared_ptr<KK> get_shared_ptr() {
            return std::shared_ptr<KK>(this);
        }
};

int main() {
    std::shared_ptr<KK> k1(new KK());
    std::shared_ptr<KK> k2 = k1->get_shared_ptr();
    std::cout << k1.use_count() << std::endl;
    std::cout << k2.use_count() << std::endl;
}

```

Good case:

```cpp
#include <iostream>
#include <memory>

//struct KK : std::enable_shared_from_this<KK> {
class KK : public std::enable_shared_from_this<KK> {
    public:
        std::shared_ptr<KK> get_shared_ptr() {
            return shared_from_this();
        }
};

int main() {
    std::shared_ptr<KK> k1(new KK());
    std::shared_ptr<KK> k2 = k1->get_shared_ptr();
    std::cout << k1.use_count() << std::endl;
    std::cout << k2.use_count() << std::endl;
}

```

Note:

`public` should be added before `class` since

```cpp
struct Test : enable_shared_from_this<Test>
```
derives **publicly** from `enable_shared_from_this`;

while

```cpp
class Test : enable_shared_from_this<Test>
```
derives **privately** from `enable_shared_from_this`;

