This project tries to find out whether the daily K-line  can be used to predict A shares. It took two months and tried a lot. Some explanations and thanks:   

* Daily K-line and minute-line data (opening price, closing price, maximum price, minimum price, turnover ratio), data from Tong Da Xin.

* some information about the company's Fundamentals (price earnings ratio, market rate, circulation stock number, total stock number). data from web.ifzq.gtimg.cn

* To extract the features of 32-day K-line images, the network directly uses MobileNetV2 and DenseNet.  

* To try to predict stocks from the time series of data of Daily K-line, referring to https://github.com/ysn2233/attentioned-dual-stage-stock-prediction.    

* To try to use the idea of game between bankers and retailers to predict stocks (based on reinforcement learning), referring to https://github.com/JannesKlaas/sometimes_deep_sometimes_learning.  

* To try to use the idea of game between bankers and retailers to predict stocks (based on Monte Carlo Tree). referring to https://github.com/junxiaosong/AlphaZero_Gomoku and the paper "Mastering the game of Go without human knowledge"

* Some of the test data was lost because of the long time.

Some tests and conclusions:
====

* A single stock and its corresponding date of the Shanghai Stock Exchange data into a row. Construct 10 1*100 column vectors for 100 consecutive days.     
   * The elements of the covariance matrix are all tending to zero, indicating that there is no correlation between the data (opening   price, closing price, maximum price, minimum price, turnover ratio), which is random.    
   
     ![](https://github.com/qjchen1972/stock/blob/master/img/covmat.png)

   * Using SVD decomposition, the maximum eigenvector is much larger than other terms. 
   
     ![](https://github.com/qjchen1972/stock/blob/master/img/singular.png)

   * The second eigenvector of the ranking corresponds to the second column of the right matrix after SVD decomposition. It can be seen that the data of stock and the data of the market are different.
   
     ![](https://github.com/qjchen1972/stock/blob/master/img/right.png)


* From the data of all A-share stocks in the last 15 years, we took 32 consecutive days of a stock and the corresponding market of that day to compose an image, and randomly selected 500,000 stocks for CNN training. The validation set did not drop. The validation set also does not decline by randomly selecting only one scenario (such as only predicting post-trough trend, or only predicting post-peak trend).  
   * [StockV1.0](https://github.com/qjchen1972/stock/tree/master/stockV1.0) uses simple five categories (x > 2%, 2% > x > 1% 1% > x >= - 1% - 1% > x >= - 2% x > - 2%). The validation set tests show that the error increases instead of decreasing.
   
     ![](https://github.com/qjchen1972/stock/blob/master/img/000001_20130604.png)
     
   
   * [StockV2.0](https://github.com/qjchen1972/stock/tree/master/stockV2.0) tries to find the best point of Auroc curve by using threshold value. The Auroc value does not exceed 0.55, which is similar to random guess.    
   
     ![](https://github.com/qjchen1972/stock/blob/master/img/stockV2_train_124.png)

   * CONCLUSION: The daily K-line can not be used regularly.


* From the data of all A-share stocks in the last 15 years, a time series is formed by taking 10 consecutive days of a stock and the corresponding data of the market on that day. 300,000 were randomly selected. RNN training using Attention mode ---[LSTM](https://github.com/qjchen1972/stock/tree/master/lstm)    
   * The validation set has dropped in 150 epoch!!   
   
     ![](https://github.com/qjchen1972/stock/blob/master/img/train_164.png)
   
   * It is surprising that the Auroc value of the validation or test set can reach 0.66-0.71. Is there any hope?      
   
   * after in-depth analysis, the main reason for the Auroc value of the validation set is close to 0.7, that is, second days after continuous the Daily-limit, the probability is very high. So the prediction accuracy is quite high. But it's no use, because you can't buy it!!!


* The essence of stock is game, can we find some rules from the perspective of game?    

   * [Img_RL](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/README.md), according to the daily K-line, carries on one of the three operations of buying, selling and holding for 10 consecutive days, and chooses the operation combination which obtains the greatest profit. Using image feature extraction or data input directly, and using the usual intensity learning training, it is found that it can not converge. Careful analysis, the reason is that the maximum revenue operation corresponding to the same characteristics is random!!!  
   
    * [Mcts](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/README.md), the previous day's rise and fall as the game operation of the banker, retailers buy, sell and hold as the corresponding game operation. So the stock becomes a game. Referring to the source code and paper of Alpha Zero of deepmind, Monte Carlo tree is used for training. If the daily K-line characteristics correspond to the rise and fall is fixed, the accuracy and return are high. Unfortunately, the rise and fall of the K-line characteristics are irregular.    
    
       ![](https://github.com/qjchen1972/stock/blob/master/img/score.png)
       
 
Dependencies
====

* pytorch V0.4.1 or later
* c++11 or later
    

