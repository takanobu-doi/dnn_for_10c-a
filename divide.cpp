#include <iostream>
#include <fstream>
#include <string>

int main()
{
  std::ifstream ifs("sca-0_tot.dat");
  std::ofstream ofs;
  std::string s;
  int i = 0;
  while(getline(ifs, s)){
    if(i%1000 == 0){
      std::string filename = "sca-0_tot-"+std::to_string(i/1000)+".dat";
      ofs.open(filename, std::ios::out);
    }
    ofs << s;
    if(i%1000 == 999){
      ofs.close();
    }
    if(i%100==0){
      std::cout << i << std::endl;
    }
    
    i++;
  }
  
  return 0;
}
