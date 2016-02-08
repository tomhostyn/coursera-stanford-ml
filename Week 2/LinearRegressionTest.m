
%%% 1 

M2 = [7921 5184 8836 4761]

scale = max (M2) - min(M2)
m = mean(M2)

SM2 = (M2 - m) / scale

SM2(2)  
 
%  -0.37388 

round(SM2(2)*100)/100  

 
%%% 2

X: 23x6
XT: 6x23

(XTxX): 6x6
(XTxX)-1xXT :6x23
 y: 23x1

theta=(XTxX)-1xXTy : 6x1
 