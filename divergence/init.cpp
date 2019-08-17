#include <iostream>

void run(int num_diverge_threads);

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: bank-conflicts <offset>" << std::endl;
        return 1;
    }

    run(std::stoi(argv[1]));

    return 0;
}
