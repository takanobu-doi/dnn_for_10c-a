#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <cmath>
#include <iomanip>

void show_progress(int &sum_num, int event_num);

int main()
{
  int sum_num = 0;
  int event_num = 5000;

  std::thread ths = std::thread([&sum_num](){
      short in;
      int r;
      int theta;
      double pi = 3.141592635897932338463;
      
      std::string s;
      std::stringstream stream;
      std::ofstream ofs_tot("sca-0_tot.dat");
      std::ofstream ofs_hough("sca-0_hough.dat");
      std::string filename[5] = {"sca-0_single_tot.dat",
				 "sca-0_more_0_tot.dat",
				 "sca-0_more_1_tot.dat",
				 "sca-0_more_2_tot.dat",
				 "sca-0_more_3_tot.dat"};
      
      for(int ifile=0;ifile<5;ifile++){
	std::ifstream ifs(filename[ifile]);
	while(getline(ifs, s)){
	  sum_num++;
	  
	  //**** hough_temp initialization ****//
	  std::vector<std::vector<std::vector<short>>> hough;
	  for(int i=0;i<2;i++){
	    std::vector<std::vector<short>> hough_temp;
	    for(int j=0;j<512;j++){
	      std::vector<short> hough_temp_temp(180);
	      hough_temp.push_back(hough_temp_temp);
	    }
	    hough.push_back(hough_temp);
	  }
	  //**** hough_temp initialization ****//
	  stream.clear();
	  stream << s;
	  for(int i=0;i<2;i++){
	    for(int j=0;j<1024;j++){
	      for(int k=0;k<256;k++){
		stream >> in;
		ofs_tot << in << " ";
		if(in == 1){
		  for(theta=0;theta<180;theta++){
		    r = (short)((sin(theta*pi/180)*sin(theta*pi/180)/cos(theta*pi/180))*k-sin(theta*pi/180)*j);
		    if(r>=0 && r<512){
		      hough[i][r][theta]++;
		    }
		  }
		}
	      }
	    }
	  }
	  ofs_tot << std::endl;
	  for(int i=0;i<2;i++){
	    for(int j=0;j<512;j++){
	      for(int k=0;k<180;k++){
		ofs_hough << hough[i][j][k] << " ";
	      }
	    }
	  }
	  ofs_hough << std::endl;
	  
	}
	ifs.close();
      }
      ofs_tot.close();
      ofs_hough.close();
    });

  show_progress(sum_num, event_num);

  ths.join();
  
  return 0;
}

void show_progress(int &sum_num, int event_num)
{
  double percent;
  unsigned int max_chr = 50;
  int step = 0;
  int num_size = log10(event_num)+1;
  std::string progress;
  time_t start = time(NULL);
  time_t pass;

  std::cout << "\e[?25l" << std::flush;

  while(percent!=100){
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    progress = "";
    percent = (sum_num*100)/(double)event_num;
    for(unsigned int ii=0;ii<(sum_num*max_chr)/event_num;ii++){
      progress += "#";
    }
    if(progress.size()<max_chr){
      switch(step){
      case 0:
	step = 1;
	progress += "-";
	break;
      case 1:
	step = 2;
	progress += "\\";
	break;
      case 2:
	step = 3;
	progress += "|";
	break;
      case 3:
	step = 0;
	progress += "/";
	break;
      default:
	step = 0;
	progress += "-";
	break;
      }
    }

    pass = time(NULL)-start;

    std::cout << "[" << "\e[1;33;49m"
	      << std::setfill(' ') << std::setw(max_chr) << std::left << progress << "\e[0m" << "]"
	      << std::setfill(' ') << std::setw(5) << std::right << std::fixed << std::setprecision(1) << percent << "%"
	      << "(" << std::setfill(' ') << std::setw(num_size) << std::right << sum_num
	      << "/" << event_num << ")"
	      << " " << pass/(60*60) << ":"
	      << std::setfill('0') << std::setw(2) << std::right << (pass/60)%60 << ":"
	      << std::setfill('0') << std::setw(2) << std::right << pass%60 << std::flush;
    if(progress.size() == max_chr && percent == 100){
      std::cout << "\e[?25h" << std::endl;
      break;
    }
//    std::cout << std::endl;
    std::cout << "\r" << std::flush;
  }
  return;
}
