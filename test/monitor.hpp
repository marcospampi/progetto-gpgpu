#include <semaphore>

using binary_semaphore = std::counting_semaphore<1>;

template < class T >
class Monitor {
public:
    Monitor( T resource ) : resource(resource), sem(1) {};
    class Helper {
    public:
        ~Helper( ) {
            monitor->sem.release();
        }
        T* operator->() {
            return &monitor->resource;
        }
        T& get() {
            return monitor->resource;
        }
    private:
        friend class Monitor;
        Monitor *monitor;
        Helper( Monitor *monitor ): monitor(monitor) {
            monitor->sem.acquire();
        }
    };
    Helper operator->() {
        return Helper(this);
    }
    Helper safe() {
        return Helper(this);
    }
    T& unsafe() {
        return resource;
    }
    void unsafeLock() {
        sem.acquire();
    }
    void unsafeRelease() {
        sem.release();
    }
private:
    friend class Helper;
    binary_semaphore sem;
    T resource;
};