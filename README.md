This project tries to find out whether the daily K-line chart can be used to predict A shares. It took two months and tried a lot. Some explanations and thanks:   

* Daily K-line and minute-line data (opening price, closing price, maximum price, minimum price, turnover ratio), data from Tong Da Xin.    * some information about the company's Fundamentals (price earnings ratio, market rate, circulation stock number, total stock number). data from web.ifzq.gtimg.cn

* To extract the features of 32-day K-line images, the network directly uses MobileNetV2 and DenseNet.  

* To try to predict stocks from the time series of data of Daily K-line, referring to https://github.com/ysn2233/attentioned-dual-stage-stock-prediction.    

* To try to use the idea of game between bankers and retailers to predict stocks (based on reinforcement learning), referring to https://github.com/JannesKlaas/sometimes_deep_sometimes_learning.  

* To try to use the idea of game between bankers and retailers to predict stocks (based on Monte Carlo Tree). referring to https://github.com/junxiaosong/AlphaZero_Gomoku and the paper "Mastering the game of Go without human knowledge"

Some attempts and conclusions:
====

* A single stock and its corresponding date of the Shanghai Stock Exchange data into a row. Construct 10 1*100 column vectors for 100 consecutive days.     
1. The elements of the covariance matrix are all tending to zero, indicating that there is no correlation between the data (opening price, closing price, maximum price, minimum price, turnover ratio), which is random.    
![](https://github.com/qjchen1972/stock/blob/master/img/000001_20130604.png)

2. Using SVD decomposition, the maximum eigenvector is much larger than other terms. 
![](https://github.com/qjchen1972/stock/blob/master/img/000001_20130604.png)

3. The second eigenvector of the ranking corresponds to the second column of the right matrix after SVD decomposition. It can be seen that the data of stock and the data of the market are different.
![](https://github.com/qjchen1972/stock/blob/master/img/000001_20130604.png)



An entertainment project, trying to get rules from historical data of stocks. it provides some features:

*  The historical data of a single stock and the corresponding data of the stock market are transformed into K-line graph, and then deep learning is used to extract the rules. 
   
   ![](https://github.com/qjchen1972/stock/blob/master/img/000001_20130604.png)
   
*  Training and predicting images are only stocks falling into the trough. 
*  confusion martix
