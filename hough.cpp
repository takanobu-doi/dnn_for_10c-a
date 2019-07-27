#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <cmath>
#include <iomanip>
#include <mutex>

void show_progress(int &sum_num, int event_num);

int main()
{
  int sum_num = 0;
  int event_num = 5000;
  std::mutex mtx;

  std::ofstream ofs_tot("sca-0_tot.dat");
  std::ofstream ofs_val("sca-0_teachervalue.dat");
  std::ofstream ofs_hough("sca-0_hough.dat");
  std::string filename[5] = {"sca-0_single",
			     "sca-0_more_0",
			     "sca-0_more_1",
			     "sca-0_more_2",
			     "sca-0_more_3"};
  std::stringstream ostream_tot[5];
  std::stringstream ostream_val[5];
  std::stringstream ostream_hough[5];

  std::vector<std::thread> ths(5);

  for(int i=0;i<5;i++){
    ths[i] = std::thread([&sum_num, &mtx](std::string filename, std::stringstream &ofs_tot, std::stringstream &ofs_val, std::stringstream &ofs_hough){
	std::ifstream ifs_tot(filename+"_tot.dat");
	std::ifstream ifs_val(filename+"_teachervalue.dat");
	short in;
	int r;
	int theta;
	double pi = 3.141592635897932338463;

	int flag = 0;
	
	std::string s;
	std::stringstream stream;
	while(getline(ifs_tot, s)){
	  mtx.lock();
	  sum_num++;
	  mtx.unlock();
	  
	  //**** hough_temp initialization ****//
	  std::vector<std::vector<std::vector<short>>> tot;
	  for(int i=0;i<2;i++){
	    std::vector<std::vector<short>> tot_temp;
	    for(int j=0;j<1024;j++){
	      std::vector<short> tot_temp_temp(256);
	      tot_temp.push_back(tot_temp_temp);
	    }
	    tot.push_back(tot_temp);
	  }
	  std::vector<std::vector<std::vector<short>>> hough;
	  for(int i=0;i<2;i++){
	    std::vector<std::vector<short>> hough_temp;
	    for(int j=0;j<1024;j++){
	      std::vector<short> hough_temp_temp(360);
	      hough_temp.push_back(hough_temp_temp);
	    }
	    hough.push_back(hough_temp);
	  }
	  //**** hough_temp initialization ****//
	  stream.str("");
	  stream.clear(std::stringstream::goodbit);
	  stream << s;
	  for(int i=0;i<2;i++){
	    for(int j=0;j<1024;j++){
	      for(int k=0;k<256;k++){
		stream >> in;
		tot[i][j][k] = in;
		if(in == 1){
		  flag = 1;
		  for(theta=0;theta<360;theta++){
		    r = (short)((sin(theta*pi/180)*sin(theta*pi/180)/cos(theta*pi/180))*k-sin(theta*pi/180)*j);
		    if(r>=0 && r<1024){
		      hough[i][r][theta]++;
		    }
		  }
		}
	      }
	    }
	  }
	  
	  for(int i=0;i<2;i++){
	    for(int j=0;j<1024;j++){
	      for(int k=0;k<256;k++){
		ofs_tot << tot[i][j][k] << " ";
	      }
	    }
	  }
	  ofs_tot << std::endl;
	  for(int i=0;i<2;i++){
	    for(int j=0;j<1024;j++){
	      for(int k=0;k<360;k++){
		ofs_hough << hough[i][j][k] << " ";
	      }
	    }
	  }
	  ofs_hough << std::endl;

	  stream.str("");
	  stream.clear(std::stringstream::goodbit);
	  getline(ifs_val, s);
	  stream << s;
	  for(int i=0;i<11;i++){
	    stream >> in;
	    ofs_val << in << " ";
	  }
	  ofs_val << std::endl;
	  
	}
	ifs_tot.close();
	ifs_val.close();
	
      },filename[i], std::ref(ostream_tot[i]), std::ref(ostream_val[i]), std::ref(ostream_hough[i]));
  }
  
  show_progress(sum_num, event_num);

  for(int i=0;i<5;i++){
    ths[i].join();
    ofs_tot << ostream_tot[i].str();
    ofs_val << ostream_val[i].str();
    ofs_hough << ostream_hough[i].str();
  }
  
  ofs_tot.close();
  ofs_val.close();
  ofs_hough.close();

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
