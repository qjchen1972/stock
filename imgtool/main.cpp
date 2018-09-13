#include"stock.h"

int main()
{
	Stock stock;
	//stock.create_serialdata("d:/stock/stock_testlabel2.txt", "d:/stock/test_stockimg2", 1, 0,1);
	stock.create_serialdata("d:/stock/max/stock_testlabel.txt", "d:/stock/max/dst", 1, 5, 6000);
	//stock.create_randtrain_data("d:/stock/stock_label.txt", "d:/stock/stock_wimg", 5, 5, 500000);
	//stock.create_randtrain_data("d:/stock/stock_testlabel.txt", "d:/stock/test_stockimg", 5, 5, 1000);
	//stock.create_data(1,0);
	//stock.create_train_data(800000);
	//stock.clear_file("d:/stock/stock_testlabel.txt");
	//stock.split("d:/stock/stock_label.txt", "d:/stock/stock_testfivelabel.txt");
	return 0;
}
