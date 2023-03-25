#include <torch/torch.h>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::cin;

int main() {
#ifdef _WIN32
    std::string data_root = R"(C:\Users\17110\Desktop\causal ad\ts.dataset\swat)";
#else
    std::string data_root=R"(/remote-home/liuwenbo/pycproj/tsdata/data)";
#endif
    cout << "data root: " << data_root << endl;
    torch::Tensor tensor = torch::eye(3);
    cout << tensor << endl;
    return 0;
}