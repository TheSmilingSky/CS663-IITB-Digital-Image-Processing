function [U, V] = MySVD(A)
    P = A*A';
    [U, ~] = eigs(P);
    Q = A'*A;
    [V, ~] = eigs(Q);
end
