#include <Windows.h>
#include <iostream>
#include <fstream>
#include<math.h>
#include<ctime>

#include "utils/netdata.h"
#include "utils/stockimage.h"
#include "utils/Tdxdata.h"
#include "utils/statics.h"
#include "dataproc.h"

void create_data() {
	DataProc  proc;
	
	proc.create_trainData(60000, 32, "20180606", "20190606");
	proc.create_testData(800, 32, "20190604", "20190628");
	proc.create_predData(800, 32, "20190606", "20190628");
}

void analyse_data(int code, std::string time,int len) {

	DataProc  proc;
	proc.proc_stock(code, time,len);
	//proc.proc_data();
}


void down_stock_info() {

	NetData net;
	AStock stock;

	net.init_Baseinfo();

	int code = 0;
	char file[128];
	while (stock.iterator(code)) {
		info_t info;
		if (!net.create_BaseInfo(stock.getLoc(code), info)) {
			printf("proc %d error \n", code);
			continue;
		}
		sprintf_s(file, "mini/company/%06d.txt", code);
		stock.write_BaseInfo(info, file);
	}
	net.exit();
}

void down_stock_kday() {

	NetData net;
	AStock stock;

	net.init_Kline();

	int code = 0;
	std::map<std::string, stock_t> Kdata;
	char file[128];
	while (stock.iterator(code)) {
		if (!net.create_Kline(stock.getLoc(code), Kdata, 2018, 2019)) {
			printf("proc %d error \n", code);
			continue;
		}
		sprintf_s(file, "mini/kday/%06d.txt", code);
		stock.write_Kline(Kdata, file);
	}
	net.exit();
 }


int main(int argc, char **argv){

	std::clock_t start = clock();
	analyse_data(603303, "20180606", 12);
	//down_stock_info();
	//down_stock_kday();
	//create_data();
	printf("time is  %f\n", float(clock() - start) / CLOCKS_PER_SEC);
	return 0;

}








