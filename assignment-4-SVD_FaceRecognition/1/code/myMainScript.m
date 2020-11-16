A = randi(1000,20,12);
[U,S,V] = MySVD(A);
disp(abs(sum(sum(A-U*S*V'))));