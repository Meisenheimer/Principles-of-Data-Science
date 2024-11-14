
#ifdef _WIN32
#include <direct.h>
#endif
#if __linux__
#include <sys/stat.h>
#endif

#include "Config.h"

std::string dir_base;

Int node = 0;
Int size = 0;
Int step = 0;
Real rho = 0;

inline void process(const std::string &id);

int main(const int argc, const char *argv[])
{
    if (argc != 5)
    {
        return 0;
    }
    node = (Int)std::atoll(argv[1]);
    size = (Int)std::atoll(argv[2]);
    step = (Int)std::atoll(argv[3]);
    rho = (Real)std::atof(argv[4]);

    dir_base = "../Data/functional_connectivity/node=" + std::to_string(node) +
               "_size=" + std::to_string(size) +
               "_step=" + std::to_string(step) +
               "_rho=" + std::to_string(rho) + "/";

#ifdef _WIN32
    mkdir("../Data/functional_connectivity/");
    mkdir(dir_base.c_str());
#endif
#if __linux__
    mkdir("../Data/functional_connectivity/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(dir_base.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

    std::fstream fp("../Data/subjectIDs.txt");
    std::vector<std::string> id_list;
    while (!fp.eof())
    {
        std::string s;
        std::getline(fp, s);
        if (s == "")
        {
            continue;
        }
        id_list.push_back(s);
    }
    fp.close();

#pragma omp parallel for
    for (Int i = 0; i < (Int)id_list.size(); i++)
    {
        process(id_list[i]);
    }

    std::cout << "node=" << node << " size=" << size << " step=" << step << " rho=" << rho << " Done.\n";

    return 0;
}

inline void process(const std::string &id)
{
    Matrix ts = loadTimeSeries(node, id);
    std::vector<Matrix> res_fc_pc;

    const Int node = ts.rows();
    const Int t = ts.cols();
    for (Int i = size - 1; i < t; i += step)
    {
        res_fc_pc.push_back(fc_pc((Matrix)ts.block(0, i - (size - 1), node, size), rho));
    }

    std::fstream fp_fc_pc(dir_base + id + ".txt", std::ios::out);
    for (Int t = 0; t < (Int)res_fc_pc.size(); t++)
    {
        const Matrix &m_fc_pc = res_fc_pc.at(t);
        bool flag = false;
        for (Int i = 0; i < node; i++)
        {
            for (Int j = i + 1; j < node; j++)
            {
                if (flag)
                {
                    fp_fc_pc << " ";
                }
                fp_fc_pc << m_fc_pc(i, j);
                flag = true;
            }
        }
        fp_fc_pc << std::endl;
    }
    fp_fc_pc.close();
    return;
}
