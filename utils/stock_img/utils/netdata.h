#ifndef _NETDATA__H
#define _NETDATA__H

#include <iostream>
#include <fstream>
#include<math.h>
#include <set>
#include <vector>
#include<string>
#include <sstream>
#include <algorithm>
#include <json/json.h>

#include "Astock.h"
#include <WinInet.h>
#pragma comment(lib, "WinInet.lib")
#pragma comment(lib, "json_vc71_libmt.lib")


class NetData {

public:
	NetData() {}
	~NetData() {
		exit();
	}	

	int init_Baseinfo() {
		return init(baseinfo_url);
	}

	int init_Kline() {
		return init(kline_url);
	}

	void exit() {
		if (m_hInternet) InternetCloseHandle(m_hInternet);
		if (m_hConnect)  InternetCloseHandle(m_hConnect);
	}


	int create_BaseInfo(std::string code, info_t &info) {

		char codeurl[128];		

		if (!m_hConnect) {
			std::cout << "Dont initital"<< std::endl;
			return 0;
		}		

		sprintf_s(codeurl, "/q=%s", code.c_str());
		HINTERNET hOpenRequest = HttpOpenRequestA(m_hConnect,
			"GET",
			codeurl, NULL, NULL, (LPCSTR*)"*/*", INTERNET_FLAG_DONT_CACHE, 1);

		BOOL bRequest = HttpSendRequestA(hOpenRequest, NULL, 0, NULL, 0);

		char szBuffer[1024] = { 0 };
		DWORD dwByteRead = 0;
		if (InternetReadFile(hOpenRequest, szBuffer, sizeof(szBuffer), &dwByteRead) && dwByteRead > 0)
			return procString((char*)szBuffer,info);
		else {
			std::cout << "requet " << code << " erro!" << std::endl;			
			return 0;
		}
		return 1;
	}
	/*
	  格式：time,open,close,high,low
	*/

	int create_Kline(std::string code, std::map<std::string, stock_t> &Kdata,int startyear, int endyear/*, std::string filepath = ""*/) {

		char codeurl[128];		

		if (!m_hConnect) {
			std::cout << "Dont initital" << std::endl;
			return 0;
		}		

		Kdata.clear();
		for (int i = startyear; i <= endyear; i++) {
			sprintf_s(codeurl, "/appstock/app/fqkline/get?_var=kline_dayhfq&param=%s,day,%d-01-01,%d-12-31,320,hfq", code.c_str(),i,i);
			printf("%d  %s%s \n", i,kline_url,codeurl);
			HINTERNET hOpenRequest = HttpOpenRequestA(m_hConnect,
				"GET",
				codeurl, NULL, NULL, (LPCSTR*)"*/*", INTERNET_FLAG_DONT_CACHE, 1);
			BOOL bRequest = HttpSendRequestA(hOpenRequest, NULL, 0, NULL, 0);
			std::string strbuf;

			char szBuffer[1025] = { 0 };
			DWORD dwByteRead = 0;		
			do {

				ZeroMemory(szBuffer, 1025);
				InternetReadFile(hOpenRequest, szBuffer, sizeof(szBuffer)-1, &dwByteRead);
				szBuffer[dwByteRead] = '\0';
				//printf(szBuffer);
				strbuf += szBuffer;
			} while (dwByteRead);

			
			if (strbuf.length() > 0)
				procKline(strbuf, code, Kdata);
			else {
				std::cout << "requet " << code << " erro!" << std::endl;				
				return 0;
			}	
		}
		return 1;
	}

private:

	int init(const char* url) {
		m_hInternet = InternetOpenA(agent, INTERNET_OPEN_TYPE_DIRECT, NULL, NULL, 0);
		if (m_hInternet == NULL)        return 0;
		m_hConnect = InternetConnectA(m_hInternet, url, INTERNET_DEFAULT_HTTP_PORT,
			NULL, NULL, INTERNET_SERVICE_HTTP, 0, 0);
		if (m_hConnect == NULL) {
			InternetCloseHandle(m_hInternet);
			return 0;
		}
		return 1;
	}

	int procKline(std::string str, std::string code, std::map<std::string, stock_t> &Kdata) {
		size_t start = str.find_first_of("{");
		std::string strJson = str.substr(start, str.length() - start);

		Json::Reader *pJsonParser = new Json::Reader();
		Json::Value tempVal;
		if (!pJsonParser->parse(strJson, tempVal)) return 0;
		Json::Value datas = tempVal["data"][code]["hfqday"];
		for (int i = 0; i < datas.size(); i++) {
			stock_t stock;
			for (int j = 0; j < datas[i].size(); j++){				
				if (datas[i][j].isString()) {
					std::string temp = datas[i][j].asString();
					if (j == 0) {
						temp.erase(std::remove(temp.begin(), temp.end(), '-'), temp.end());
						stock.time = temp;
					}
					else if (j == 1) stock.open = std::stof(datas[i][j].asString());					
					else if (j == 2 )   stock.close = std::stof(datas[i][j].asString());
					else if (j == 3 )   stock.high = std::stof(datas[i][j].asString());
					else if (j == 4 )   stock.low = std::stof(datas[i][j].asString());
					else if (j == 5 )   stock.vol = std::stof(datas[i][j].asString());
				}
			}
			if (!stock.time.empty()) {
				stock.code = code;
				Kdata[stock.time] = stock;
			}
		}
		return 1;
	}

	/*
		1:  股票名称
		2:  股票代码
		3:  当前价格
		36: 成交手数
		38: 换手率
		39: 市盈率
		43: 流通市值
		44: 总市值
		45: 市净率

		写入文件
		股票代码   每股收益(nowprice/市盈率)  每股净资产(nowprice/市净率)  流通股(流通市值/nowprice)   总股数(总市值/nowprice)  股票名称
	*/
	int procString(char* buf, info_t &info) {

		std::string str = buf;
		size_t start = str.find_first_of("\"");
		size_t end = str.find_last_of("\"");

		std::string temp = str.substr(start + 1, end - start - 1);
		std::vector<std::string> ans;

		if (my_split(temp, '~', ans) < 0) {
			std::cout <<" split string error" << std::endl;
			return 0;
		}
	
		float sy = 0;
		if (atof(ans[39].c_str()) > 0.01) sy = atof(ans[3].c_str()) / atof(ans[39].c_str());
		float zc = 0;
		if (atof(ans[46].c_str()) > 0.01) zc = atof(ans[3].c_str()) / atof(ans[46].c_str());
		float market = 0;
		if (atof(ans[3].c_str()) > 0.01) market = atof(ans[44].c_str()) / atof(ans[3].c_str());
		else
		{
			std::cout << ans[2] << "  data is abnormal" << std::endl;
			return 0;
		}
		float all = 0;
		if (atof(ans[3].c_str()) > 0.01) all = atof(ans[45].c_str()) / atof(ans[3].c_str());

		info.code = ans[2];
		info.sy = sy;
		info.zc = zc;
		info.tradable_shares = market;
		info.total_shares = all;
		info.name = ans[1];
		return 1;
	}

	int my_split(const std::string& src, const char& delim, std::vector<std::string>& vec)
	{
		int src_len = src.length();
		int find_cursor = 0;
		int read_cursor = 0;

		if (src_len <= 0) return -1;

		vec.clear();
		while (read_cursor < src_len) {

			find_cursor = src.find(delim, find_cursor);

			//1.找不到分隔符
			if (-1 == find_cursor) {
				if (read_cursor <= 0) return -1;

				//最后一个子串, src结尾没有分隔符
				if (read_cursor < src_len) {
					vec.push_back(src.substr(read_cursor, src_len - read_cursor));
					return 0;
				}
			}
			//2.有连续分隔符的情况
			else if (find_cursor == read_cursor) {
				//字符串开头为分隔符, 也按空子串处理, 如不需要可加上判断&&(read_cursor!=0)
				vec.push_back(std::string(""));
			}
			//3.找到分隔符
			else
				vec.push_back(src.substr(read_cursor, find_cursor - read_cursor));

			read_cursor = ++find_cursor;
			if (read_cursor == src_len) {
				//字符串以分隔符结尾, 如不需要末尾空子串, 直接return
				vec.push_back(std::string(""));
				return 0;
			}
		}//end while()

		return 0;
	}	

	const char* agent = "Microsoft InternetExplorer";
	const char* baseinfo_url = "qt.gtimg.cn";
	const char* kline_url = "web.ifzq.gtimg.cn";
	HINTERNET m_hInternet = NULL;
	HINTERNET m_hConnect = NULL;
};

#endif