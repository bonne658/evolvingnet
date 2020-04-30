#include <Net.h>

int main() {
    Net n;
    n.Train();
    n.Test(0.9, 0.01);
    return 0;
}
