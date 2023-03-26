#include <torch/torch.h>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::cin;

int main() {
#ifdef _WIN32
//    const std::string data_root = R"(C:\Users\17110\Desktop\causal ad\ts.dataset\swat\)";
    const std::string data_root=R"(E:\Pycharm Projects\causal.dataset\data\swat\)";
#else
    const std::string data_root=R"(/remote-home/liuwenbo/pycproj/tsdata/data/)";
#endif
    const std::string train_set_file="swat.att.n0a1.csv";
    const std::string test_set_file="swat.nor.csv";
    std::ifstream input_file(data_root+train_set_file);
    int line_num=0,column_num=0;
    for (std::string line_buffer; std::getline(input_file, line_buffer); )
    {
        std::cout << line_buffer << std::endl;
        line_num++;
    }
    torch::Tensor tensor = torch::eye(3);
    cout << tensor << endl;
    return 0;
}